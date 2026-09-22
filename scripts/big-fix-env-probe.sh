#!/usr/bin/env bash
# Phase-0 environment probe for the big-fix plan (.omo/plans/big-fix.md, todo 1).
#
# Read-only host inspection: no network, no GPU work, no builds, and it never
# deletes or mutates anything inside an existing build-cuda/ directory (stat only).
# Idempotent: safe to rerun; the evidence file is fully rewritten each run.
#
# Writes evidence to .omo/evidence/big-fix/env-probe.txt and mirrors it to stdout.
# Exit policy:
#   0  -> required_cpu_lane: pass
#   1  -> required_cpu_lane: blocked (missing required tool; report a toolchain
#         blocker; do NOT mark any todo after todo 1 complete)
# Required CPU lane tools: cmake, ninja (generator), ctest, Qt6, eigen3.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVIDENCE="${REPO_ROOT}/.omo/evidence/big-fix/env-probe.txt"

mkdir -p "$(dirname "${EVIDENCE}")" || {
    echo "probe_error: cannot create evidence dir $(dirname "${EVIDENCE}")" >&2
    echo "required_cpu_lane: blocked"
    exit 1
}

# Bounded execution helper: every external command gets a hard timeout so the
# probe can never hang on a stuck toolchain binary.
run_bounded() {
    if command -v timeout >/dev/null 2>&1; then
        timeout 10 "$@" 2>&1
    else
        "$@" 2>&1
    fi
}

# stdin VERSION_OUTPUT -> single-line, trimmed version string
first_line() { grep -m1 '[^[:space:]]' | cut -c1-120 | tr -d '\r'; }

# probe_binary LABEL BIN [ARG...] -> "LABEL: <version>" or "LABEL: missing"
probe_binary() {
    local label="$1" bin="$2"; shift 2
    if ! command -v "$bin" >/dev/null 2>&1; then
        printf '%s: missing\n' "$label"
        return 1
    fi
    local ver
    ver="$(run_bounded "$bin" "$@" | first_line || true)"
    if [ -n "$ver" ]; then
        printf '%s: %s (%s)\n' "$label" "$ver" "$bin"
    else
        printf '%s: present but version check failed (%s)\n' "$label" "$bin"
    fi
}

probe_pkgconfig() {
    local label="$1" mod="$2"
    if ! command -v pkg-config >/dev/null 2>&1; then
        printf '%s: missing (pkg-config unavailable)\n' "$label"
        return 1
    fi
    if run_bounded pkg-config --exists "$mod" ; then
        printf '%s: %s (pkg-config %s)\n' "$label" "$(run_bounded pkg-config --modversion "$mod" | first_line)" "$mod"
        return 0
    fi
    printf '%s: missing\n' "$label"
    return 1
}

# Qt6: prefer pkg-config Qt6Core; fall back to qmake6; a Qt5-only qmake is NOT Qt6.
probe_qt6() {
    if command -v pkg-config >/dev/null 2>&1 && run_bounded pkg-config --exists Qt6Core; then
        printf 'Qt6: %s (pkg-config Qt6Core)\n' "$(run_bounded pkg-config --modversion Qt6Core | first_line)"
        return 0
    fi
    if command -v qmake6 >/dev/null 2>&1; then
        printf 'Qt6: %s (qmake6)\n' "$(run_bounded qmake6 -query QT_VERSION | first_line)"
        return 0
    fi
    if command -v qmake >/dev/null 2>&1; then
        local v
        v="$(run_bounded qmake -query QT_VERSION | first_line)"
        case "$v" in
            6.*) printf 'Qt6: %s (qmake)\n' "$v"; return 0 ;;
            *)   printf 'Qt6: missing (qmake found but QT_VERSION=%s, not Qt6)\n' "${v:-unknown}" ; return 1 ;;
        esac
    fi
    printf 'Qt6: missing\n'
    return 1
}

# Optional: libfreenect via pkg-config, else distro header location.
probe_libfreenect() {
    if command -v pkg-config >/dev/null 2>&1 && run_bounded pkg-config --exists libfreenect; then
        printf 'libfreenect: %s (pkg-config libfreenect)\n' "$(run_bounded pkg-config --modversion libfreenect | first_line)"
        return 0
    fi
    if [ -f /usr/include/libfreenect/freenect.h ] || [ -f /usr/local/include/libfreenect/freenect.h ]; then
        printf 'libfreenect: present (header found, version unknown)\n'
        return 0
    fi
    printf 'libfreenect: missing\n'
    return 1
}

# GPU presence: passive inspection only (lspci / sysfs), never initializes a context.
probe_gpu() {
    local found=""
    if command -v lspci >/dev/null 2>&1; then
        found="$(run_bounded lspci -nn | grep -Ei 'VGA compatible controller|3D controller|Display controller' | head -2 | tr '\n' ';' | cut -c1-200)"
    fi
    if [ -n "$found" ]; then
        printf 'gpu: present (%s)\n' "$found"
        return 0
    fi
    if compgen -G '/dev/nvidia[0-9]*' >/dev/null || [ -e /dev/kfd ] || compgen -G '/sys/class/drm/card[0-9]*' >/dev/null 2>&1; then
        printf 'gpu: present (device node found, lspci info unavailable)\n'
        return 0
    fi
    if ! command -v lspci >/dev/null 2>&1 && ! compgen -G '/sys/class/drm/card*' >/dev/null 2>&1; then
        printf 'gpu: unknown (no lspci, no /sys/class/drm)\n'
        return 1
    fi
    printf 'gpu: absent\n'
    return 1
}

collect_report() {
    local missing_required=""

    printf '=== big-fix phase-0 environment probe ===\n'
    printf 'timestamp: %s\n' "$(date -Is 2>/dev/null || date)"
    printf 'host: %s\n' "$(uname -sr 2>/dev/null || echo unknown)"

    # Required CPU-lane tools (version or missing).
    probe_binary cmake  cmake  --version        || missing_required="${missing_required} cmake"
    probe_binary ninja  ninja  --version        || missing_required="${missing_required} ninja"
    probe_binary ctest  ctest  --version        || missing_required="${missing_required} ctest"
    probe_qt6                                   || missing_required="${missing_required} Qt6"
    probe_pkgconfig eigen3 eigen3               || missing_required="${missing_required} eigen3"

    # Optional tools / libraries.
    probe_libfreenect || true
    probe_binary nvcc nvcc --version || true
    probe_binary hipcc hipcc --version || true

    probe_gpu || true

    # Existing build-cuda/: stat only. NEVER delete, clean, or mutate it.
    if [ -d "${REPO_ROOT}/build-cuda" ]; then
        printf 'build_cuda: present\n'
        printf 'build_cuda_warning: DO NOT delete, clean, reconfigure, or otherwise mutate build-cuda/; use fresh *-big-fix build dirs for QA\n'
    else
        printf 'build_cuda: absent\n'
    fi

    if [ -z "$missing_required" ]; then
        printf 'required_cpu_lane: pass\n'
    else
        printf 'required_cpu_lane: blocked\n'
        printf 'blocked_reason: missing required tool(s):%s\n' "$missing_required"
    fi
}

TMPREPORT="$(mktemp)"
trap 'rm -f "${TMPREPORT}"' EXIT

collect_report > "${TMPREPORT}"

# Exit code derives from the freshly generated report (never from a stale file).
tee "${EVIDENCE}" < "${TMPREPORT}" >/dev/null
cat "${EVIDENCE}"
if grep -qx 'required_cpu_lane: pass' "${TMPREPORT}"; then
    exit 0
else
    printf 'TOOLCHAIN BLOCKER: required CPU test lane is blocked; this is an environment gate, not code success. Stop after todo 1.\n' >&2
    exit 1
fi
