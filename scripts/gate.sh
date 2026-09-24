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
# CPU budget. The whole gate (compilers, ctest, every test) is pinned with
# taskset to one set of CPUs. The default is the last L3 domain (on the dev
# host's 5950X: CCD1 = cores 8-15 = CPUs 8-15,24-31), because interactive work
# is pinned to CCD0 there; `--cpu-list LIST` (taskset syntax) picks another set.
# OMP_NUM_THREADS (--omp) defaults to the size of the set, which is what an
# OpenMP test would pick by itself inside the pin; ctest runs a quarter of the
# set's size in parallel (4 on 16 CPUs). The timing contracts need both: with 4
# OpenMP threads a Debug spin frame took > 3 s and the fakenect smoke test
# integrated 5 fps; with 8 tests in parallel on 16 CPUs the room-preset spin
# lost track at 121 deg (the model lagged the tracker; it passes alone and at 4).
# -j (the build) defaults to the size of the set.
# OMP_WAIT_POLICY defaults to PASSIVE: with 4 tests of 16 OpenMP threads each
# on 16 CPUs, libgomp's spinning threads (it only throttles the spin when ONE
# process has more threads than CPUs) starved each other: 4 concurrent
# relocalizer_contract runs took 210 s with the default policy, 36 s passive
# (alone: 13.2 s default, 11.3 s passive), and a Debug run timed out at 240 s.
#
# usage: scripts/gate.sh [--cpu-list LIST] [--omp N] [-j N]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPU_LIST="${GATE_CPU_LIST:-}"
JOBS=""
OMP=""
ARGS=("$@")
while [ $# -gt 0 ]; do
    case "$1" in
        --cpu-list) CPU_LIST="$2"; shift 2 ;;
        --omp)      OMP="$2"; shift 2 ;;
        -j)         JOBS="$2"; shift 2 ;;
        *)          echo "usage: scripts/gate.sh [--cpu-list LIST] [--omp N] [-j N]" >&2; exit 2 ;;
    esac
done
if [ -z "${CPU_LIST}" ]; then
    # CPUs of the highest-numbered L3 (lscpu -p: "cpu,l1d:l1i:l2:l3").
    CPU_LIST="$(lscpu -p=CPU,CACHE 2>/dev/null | grep -v '^#' \
                | awk -F'[,:]' '{ l3[$1] = $NF; if ($NF > max) max = $NF }
                                END { for (c in l3) if (l3[c] == max) print c }' \
                | sort -n | paste -sd, -)"
fi
[ -n "${CPU_LIST}" ] || CPU_LIST="0-$(( $(nproc) - 1 ))"
N_CPUS=0
IFS=, read -ra _parts <<< "${CPU_LIST}"
for _p in "${_parts[@]}"; do
    if [[ "${_p}" == *-* ]]; then N_CPUS=$(( N_CPUS + ${_p#*-} - ${_p%-*} + 1 )); else N_CPUS=$(( N_CPUS + 1 )); fi
done
JOBS="${JOBS:-${N_CPUS}}"
OMP="${OMP:-${N_CPUS}}"
CTEST_JOBS=$(( N_CPUS / 4 > 0 ? N_CPUS / 4 : 1 ))
export OMP_NUM_THREADS="${OMP}"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"

if [ -z "${GATE_PINNED:-}" ] && command -v taskset >/dev/null 2>&1; then
    echo "== pinned to ${N_CPUS} CPUs (${CPU_LIST}), -j ${JOBS}, ctest -j ${CTEST_JOBS}, OMP_NUM_THREADS=${OMP}, OMP_WAIT_POLICY=${OMP_WAIT_POLICY}"
    export GATE_PINNED=1
    exec taskset -c "${CPU_LIST}" "$0" "${ARGS[@]}"
fi

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
    if (cd "${dir}" && ctest -j "${CTEST_JOBS}" --timeout 300 >>"${log}" 2>&1); then
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
