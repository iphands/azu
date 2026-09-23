#!/usr/bin/env bash
# big-fix-two gate (T0.1): build and test every lane this host can build.
#
#   cpu-release   configure + build + ctest (the lane users run)
#   cpu-debug     configure + build + ctest
#   cuda-release  configure + build the app, when nvcc exists (no GPU ctest
#                 lane yet: tests/ is only configured for GPU_BACKEND=CPU)
#
# Build trees are build-gate-<lane>/ next to this script's repo (ignored by
# .gitignore's build-*/). They are reused, not wiped, so reruns are incremental.
#
# usage: scripts/gate.sh [-j N]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS="$(( $(nproc) > 24 ? 24 : $(nproc) ))"
if [ "${1:-}" = "-j" ] && [ -n "${2:-}" ]; then JOBS="$2"; fi

# libfreenect / libfakenect are installed under /usr/local on the dev host.
export PKG_CONFIG_PATH="/usr/local/lib64/pkgconfig:/usr/local/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

declare -a SUMMARY=()
FAILED=0

run_lane() {   # name build_type backend run_tests
    local name="$1" type="$2" backend="$3" tests="$4"
    local dir="${ROOT}/build-gate-${name}"
    local log="${dir}/gate.log"
    local testing=OFF
    [ "${tests}" = yes ] && testing=ON
    mkdir -p "${dir}"
    echo "== ${name}: configure/build (${type}, GPU_BACKEND=${backend})"
    if ! { cmake -S "${ROOT}" -B "${dir}" -G Ninja -DCMAKE_BUILD_TYPE="${type}" \
               -DGPU_BACKEND="${backend}" -DBUILD_TESTING="${testing}" \
           && cmake --build "${dir}" -j "${JOBS}"; } >"${log}" 2>&1; then
        SUMMARY+=("${name}: BUILD FAILED (see ${log})")
        FAILED=1
        return
    fi
    if [ "${tests}" != yes ]; then
        SUMMARY+=("${name}: built")
        return
    fi
    echo "== ${name}: ctest"
    if (cd "${dir}" && ctest -j 8 --timeout 300 >>"${log}" 2>&1); then
        SUMMARY+=("${name}: $(grep -E 'tests passed' "${log}" | tail -1)")
    else
        SUMMARY+=("${name}: TESTS FAILED -- $(grep -E 'tests passed' "${log}" | tail -1) (see ${log})")
        FAILED=1
    fi
}

run_lane cpu-release Release CPU yes
run_lane cpu-debug   Debug   CPU yes
if command -v nvcc >/dev/null 2>&1; then
    run_lane cuda-release Release CUDA no
else
    SUMMARY+=("cuda-release: skipped (no nvcc)")
fi

echo
echo "== gate summary"
for line in "${SUMMARY[@]}"; do echo "  ${line}"; done
exit "${FAILED}"
