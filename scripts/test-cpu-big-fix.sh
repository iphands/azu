#!/usr/bin/env bash
# CPU-only QA gate for the big-fix plan (.omo/plans/big-fix.md, todo 3).
#
# Scope contract (plan constraints D1-D5, "Non-negotiable rules"):
#   - The ONLY backend this gate configures, builds, and tests is CPU.
#   - It never runs nvcc or hipcc, never passes GPU_BACKEND=CUDA/HIP, never
#     creates or touches build-cuda-big-fix / build-hip-big-fix, and never
#     mutates the pre-existing build-cuda/ or build-cpu/ directories.
#   - Forbidden backend names appear below only as DATA: comments and the
#     documentary deferral matrix. They are never a command word or an argument
#     to cmake/ctest/ninja/nm/rm. scripts/static-check no-backend invocations
#     in the todo 3 evidence proves that mechanically.
#   - Backend evidence here is documentary only: deferred / not compiled /
#     not runtime-tested. It is never reported as compile-pass or runtime-pass.
#
# Exit status is the failing command's own status (124 on timeout); success is 0.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
GATE_BUILD_DIRNAME="build-cpu-big-fix"
DOSSIER="${REPO_ROOT}/docs/CUDA_HIP_DEFERRED_CHANGES.md"
PROBE="${REPO_ROOT}/scripts/big-fix-env-probe.sh"
CPU_LABEL="cpu"
# Required CPU tests registered by tests/CMakeLists.txt. CTest 4.3 exits 0 with
# "No tests were found!!!", so a green run without these names is NOT a pass.
# big-fix todo 12 appended icp_weighting_contract and todo 13 appended
# icp_numeric_policy_contract; todo 14 appended tsdf_integration_race_contract; todo 15
# appended the three TSDF raycast/depth-band contracts; todo 16 appended
# marching_cubes_frontier_contract; todo 17 appended the three CPU mesh-invariant
# contracts (mesh_welding_contract, mesh_color_invariant_contract,
# mesh_truncation_progress_contract); todo 18 appended
# marching_cubes_winding_contract; todo 19 appended color_convergence_contract and
# glb_linear_color_contract; todo 20 appended depth_domain_contract and
# depth_ema_determinism_contract; todo 21 appended cas_border_contract; todo 22
# appended sr_upscaled_contract; todo 23 appended kinect_pairing_contract; todo 24
# appended pipeline_hyperparams_contract; todo 25 appended
# pipeline_state_contract and pipeline_mesh_cadence_contract (29 CPU tests in
# total); todo 26 INTENTIONALLY appended pipeline_export_mesh_contract as the
# 30th CPU test (the shared exportMesh helper behind the non-blocking GUI
# export); todo 27 INTENTIONALLY appended camera_basis_contract as the 31st CPU
# test (the real src/rendering/Camera.cpp orbit/pan/basis math plus the pure
# first-frame tick clamp, which needs no Qt, display or OpenGL context); todo 29
# INTENTIONALLY appended fusion_ui_validation_contract as the 32nd CPU test (the
# real Qt-free src/gui/FusionUiModel.cpp cross-field hyperparameter validation,
# atomic preset staging and metrics style-band state machine, which needs no Qt
# Widgets, display or OpenGL context); todo 30 INTENTIONALLY appended
# ply_writer_contract as the 33rd CPU test (the real src/export/PLYExporter.cpp
# binary/ASCII schema, explicit little-endian bytes, header/payload count
# agreement, determinism and every no-file failure path, driven through both
# public write entry points and parsed off disk); todo 31 INTENTIONALLY appended
# glb_writer_contract as the 34th CPU test (the real src/export/GLBExporter.cpp
# written to disk, its GLB container and JSON chunk parsed independently, its
# accessors/buffer views/bounds/unit normals/linear COLOR_0 checked against the glTF
# spec, a tinygltf round trip over the same bytes, every invalid mesh refused with no
# file, and a real post-open write fault leaving no partial GLB); every name is
# enforced as registered (the list only ever grows, so the guard never weakens).
REQUIRED_TESTS=("cpu_smoke_harness" "pipeline_test_seam_smoke" "pipeline_stop_contract"
    "marching_cubes_table_contract" "marching_cubes_sphere_contract" "tsdf_reset_contract"
    "mesh_validation_contract" "coordinate_rounding_contract" "icp_weighting_contract"
    "icp_numeric_policy_contract" "tsdf_integration_race_contract"
    "tsdf_raycast_contract" "tsdf_integration_min_depth_contract"
    "tsdf_subvoxel_thin_feature_contract" "marching_cubes_frontier_contract"
    "mesh_welding_contract" "mesh_color_invariant_contract"
    "mesh_truncation_progress_contract" "marching_cubes_winding_contract"
    "color_convergence_contract" "glb_linear_color_contract"
    "depth_domain_contract" "depth_ema_determinism_contract" "cas_border_contract"
    "sr_upscaled_contract" "kinect_pairing_contract"
    # todo 24: live hyperparameter propagation + callback/thread serialization.
    "pipeline_hyperparams_contract"
    # todo 25: reset/start motion model + bounded-queue backpressure + versioned
    # mesh request cadence.
    "pipeline_state_contract" "pipeline_mesh_cadence_contract"
    # todo 26: shared exportMesh(path, writer_fn) helper behind exportPLY/exportGLB.
    "pipeline_export_mesh_contract"
    # todo 27: real Camera orbit/pan/basis math + the pure first-frame tick clamp.
    "camera_basis_contract"
    # todo 29: Qt-free fusion UI validation / preset staging / metrics band model.
    "fusion_ui_validation_contract"
    # todo 30: real PLY writer schema/LE bytes + every no-file failure path.
    "ply_writer_contract"
    # todo 31: real GLB writer container/glTF validity, unit normals, linear COLOR_0.
    "glb_writer_contract")
# Real controller coupling, proven post-build: a source-only replica that
# re-declares its own look-alike seam methods has none of these symbols. Both
# real-controller seam binaries must carry them...
NM_POSITIVE_TARGETS=("tests/pipeline_test_seam_smoke" "tests/pipeline_stop_contract"
    "tests/pipeline_hyperparams_contract" "tests/pipeline_state_contract"
    "tests/pipeline_mesh_cadence_contract" "tests/pipeline_export_mesh_contract")
# ...while a Qt-free control target (cpu_smoke_harness links only azu_test_core,
# never the controller TU) must NOT carry the controller's internal symbol.
NM_NEGATIVE_TARGET="tests/cpu_smoke_harness"
NM_REQUIRED_PATTERNS=("kfusion::app::PipelineController" "PipelineController::startInternal")
NM_ABSENT_PATTERN="PipelineController::startInternal"

# All scratch lives outside the repository root; nothing is written under it
# except the single validated gate build directory.
WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/bigfix-cpu-gate.XXXXXX")"
trap 'rm -rf "${WORKDIR}"' EXIT

die() {
    printf 'CPU GATE FAIL: %s\n' "$1" >&2
    exit "${2:-1}"
}

# Bounded execution + preserved exit status + command context on failure.
run_step() {
    local limit="$1" label="$2"
    shift 2
    printf '\n$ [timeout %ss] %s\n' "$limit" "$*"
    local status=0
    timeout "$limit" "$@" || status=$?
    if [ "$status" -ne 0 ]; then
        printf 'CPU GATE FAIL: %s exited %s (124 = timeout)\n' "$label" "$status" >&2
        exit "$status"
    fi
    printf '==> %s: ok\n' "$label"
}

# ---------------------------------------------------------------------------
# 0. Repo root + gate build directory validation
# ---------------------------------------------------------------------------
[ -d "${REPO_ROOT}/.git" ] || [ -f "${REPO_ROOT}/CMakeLists.txt" ] ||
    die "resolved REPO_ROOT ${REPO_ROOT} is not the azu repository root"
[ -f "${DOSSIER}" ] || die "deferral dossier ${DOSSIER} is missing; the CPU gate documents backend deferral through it"
[ -f "${PROBE}" ] || die "environment probe ${PROBE} is missing; the CPU lane cannot be gated without it"

GATE_BUILD_DIR="$(realpath -m -- "${REPO_ROOT}/${GATE_BUILD_DIRNAME}")" ||
    die "cannot resolve the gate build directory path"
# Fail-closed guards: the path this script may delete is exactly one directory,
# identified by basename AND by being a direct child of the repository root.
[ "$(basename "${GATE_BUILD_DIR}")" = "${GATE_BUILD_DIRNAME}" ] ||
    die "refusing to proceed: gate build dir basename is not '${GATE_BUILD_DIRNAME}' (got ${GATE_BUILD_DIR})"
[ "$(dirname "${GATE_BUILD_DIR}")" = "${REPO_ROOT}" ] ||
    die "refusing to proceed: gate build dir ${GATE_BUILD_DIR} is not a direct child of ${REPO_ROOT}"
printf 'repo root: %s\ngate build dir: %s\n' "${REPO_ROOT}" "${GATE_BUILD_DIR}"

# ---------------------------------------------------------------------------
# 1. Environment gate: the phase-0 probe must report the required CPU lane pass
# ---------------------------------------------------------------------------
PROBE_OUT="${WORKDIR}/probe.txt"
probe_status=0
timeout 60 bash "${PROBE}" >"${PROBE_OUT}" 2>&1 || probe_status=$?
grep -qx 'required_cpu_lane: pass' "${PROBE_OUT}" ||
    die "phase-0 probe did not report 'required_cpu_lane: pass' (probe exit ${probe_status}); see ${PROBE}"
printf 'phase-0 probe: required_cpu_lane: pass\n'

# ---------------------------------------------------------------------------
# 2. Backend deferral matrix (documentary; the acceptance artifact for todo 3)
# ---------------------------------------------------------------------------
emit_deferral_matrix() {
    printf '\n=== backend deferral matrix (documentary, this run) ===\n'
    printf 'backend: cuda\nstatus: deferred\ncompiled: not compiled\nruntime_tested: not runtime-tested\n'
    printf 'current_gate: CPU-only; CUDA compilation/runtime deferred\n'
    printf 'future_gate: DO NOT RUN IN THIS PLAN - build-cuda-big-fix gate\n\n'
    printf 'backend: hip\nstatus: deferred\ncompiled: not compiled\nruntime_tested: not runtime-tested\n'
    printf 'current_gate: CPU-only; HIP compilation/runtime deferred\n'
    printf 'future_gate: DO NOT RUN IN THIS PLAN - build-hip-big-fix gate\n'
}
emit_deferral_matrix

# ---------------------------------------------------------------------------
# 3. Dossier structure gate: the runner fails if the scaffold loses its shape
# ---------------------------------------------------------------------------
# grep -xF, not substring: a substring match let one future-gate line be deleted
# while the gate still passed (caught by todo 3 negative QA).
#
# Input form is a here-string (<<<), NOT `printf '%s\n' "$dossier" | grep`.
# Under `set -euo pipefail`, a dossier larger than the ~64 KiB pipe buffer makes
# `grep -q` match and exit before `printf` finishes writing: `printf` then takes
# SIGPIPE (141) and pipefail turns a SUCCESSFUL match into a gate failure. The
# here-string is spooled to a file by bash, so there is no producer to SIGPIPE
# and the exact-line/phrase/row-count semantics are unchanged. Do not "simplify"
# these back into a pipe; see .omo/evidence/big-fix/gate-sigpipe-fix.txt.
dossier="$(cat "${DOSSIER}")"
require_fixed_line() {
    local needle="$1"
    grep -qxF -- "$needle" <<< "${dossier}" ||
        die "dossier ${DOSSIER} is missing the required exact line: ${needle}"
}
DOSSIER_EXACT_LINES=(
    '# CUDA/HIP deferred changes from big-fix CPU-only execution'
    'Status: deferred; not compiled in this run'
    '## Environment at deferral time'
    '## Future backend gate'
    '## Deferred change table'
    '| Audit ID | Backend | File/function | Current CPU decision | Required CUDA/HIP change | Risk if not done | Future acceptance |'
    '- DO NOT RUN IN THIS PLAN: cmake -S . -B build-cuda-big-fix -G Ninja -DGPU_BACKEND=CUDA -DBUILD_TESTING=OFF'
    '- DO NOT RUN IN THIS PLAN: cmake --build build-cuda-big-fix'
    '- DO NOT RUN IN THIS PLAN: cmake -S . -B build-hip-big-fix -G Ninja -DGPU_BACKEND=HIP -DBUILD_TESTING=OFF'
    '- DO NOT RUN IN THIS PLAN: cmake --build build-hip-big-fix'
)
for dossier_line in "${DOSSIER_EXACT_LINES[@]}"; do
    require_fixed_line "$dossier_line"
done
for backend_phrase in deferred 'not compiled' 'not runtime-tested'; do
    grep -qF -- "$backend_phrase" <<< "${dossier}" ||
        die "dossier ${DOSSIER} is missing required backend status phrase: ${backend_phrase}"
done
dossier_rows="$(grep -c '^| `' <<< "${dossier}" || true)"
[ "${dossier_rows}" -ge 1 ] || die "dossier has no deferred-change rows"
printf 'dossier gate: ok (%s deferred-change rows)\n' "${dossier_rows}"

# ---------------------------------------------------------------------------
# 3b. CPU dead-symbol grep gate: the absent-symbol checks below are a standing
#     contract; any occurrence (code or comment) of a listed pattern anywhere
#     in the CPU-owned scope fails the gate. Scope is CPU-owned source only: *.cpp / *.h / CMakeLists.txt under src/,
#     include/, tests/ and the root CMakeLists.txt. Backend translation units
#     (*.cu / *.hip / *.cuh) and include/tsdf/VoxelGPU.h are excluded twice
#     over: the include globs never match them and a defensive filter drops
#     them from any hit list. Deferred BACKEND dead code (dossier rows
#     cross-backend:A34 syncFromGPU/getGPUDepthRaw, cross-backend:A35,
#     cross-backend:B2 ICPParams::min_depth/max_depth, tsdf:T18 integrateGPU
#     R_cw/t_cw, tsdf:T25 VoxelGPU.h ALIGN32) is deliberately NOT scanned: it
#     lives in backend files or backend-guarded regions and only a future
#     compiled backend lane may remove it. This section never invokes any
#     backend; grep is its only tool.
# ---------------------------------------------------------------------------
dead_symbol_absent_check() {
    local label="$1" pattern="$2"
    shift 2
    local -a globs=("$@")
    [ "${#globs[@]}" -gt 0 ] || globs=('*.cpp' '*.h' 'CMakeLists.txt')
    local -a includes=()
    local g
    for g in "${globs[@]}"; do includes+=(--include="${g}"); done
    local hits
    hits="$(grep -rnE "${includes[@]}" -e "${pattern}" \
        "${REPO_ROOT}/src" "${REPO_ROOT}/include" "${REPO_ROOT}/tests" \
        "${REPO_ROOT}/CMakeLists.txt" 2>/dev/null |
        grep -vE '(^|/)VoxelGPU\.h:|_cuda\.cu:|_hip\.hip:|\.cuh:' || true)"
    if [ -n "${hits}" ]; then
        printf '%s\n' "${hits}" >&2
        die "dead-symbol gate: ${label} reintroduced in CPU scope (pattern: ${pattern})"
    fi
    printf 'dead-symbol gate: absent in CPU scope: %s (%s)\n' "${label}" "${pattern}"
}
dead_symbol_absent_check 'tracking:CPU-7 model-projection member' 'projectModel'
dead_symbol_absent_check 'tracking:CPU-8 level-scaled intrinsic helpers' '\bget(Fx|Fy|Cx|Cy)\b'
dead_symbol_absent_check 'gui:GL-23 dead preview projection' 'sensorProjection'
dead_symbol_absent_check 'gui:GL-29 unused slider widget' 'QSlider' 'ControlPanel.*'
dead_symbol_absent_check 'pipeline:PC-26 metrics callback plumbing + UI skip counter' \
    'MetricsCallback|metrics_cb_|setMetricsCallback|ui_skip_counter_'
dead_symbol_absent_check 'pipeline:PC-20 no-op frame release helper' 'releaseData'
dead_symbol_absent_check 'pipeline:PC-16 always-true modulo' '% 1 == 0' 'PipelineController.cpp'
dead_symbol_absent_check 'app:JS-01/RB-01..03 unused JobSystem + RingBuffer' 'RingBuffer|JobSystem'
dead_symbol_absent_check 'app:JS-04 removed std::result_of usage' 'std::result_of'
dead_symbol_absent_check 'meshing:D6 dead triangle struct + adder' 'struct Triangle|addTriangle'
dead_symbol_absent_check 'meshing:D4 dead final-mesh member' 'mesh_final'
dead_symbol_absent_check 'meshing duplicate hashers (todo 17 lineage)' 'VectorHash|VertexHasher'
dead_symbol_absent_check 'tsdf:T24 false bounds-checked doc claim' 'bounds checked' 'TSDFVolume.*'

# Backend-file tripwire: this plan is CPU-only, so the working tree (staged or
# unstaged, relative to HEAD) must never touch a backend TU or the GPU voxel
# header. Vacuously true after a clean commit; its teeth are during development.
backend_touches="$(git -C "${REPO_ROOT}" diff --name-only HEAD 2>/dev/null |
    grep -E '\.(cu|hip|cuh)$|VoxelGPU\.h$' || true)"
[ -z "${backend_touches}" ] ||
    die "CPU-only plan constraint violated: backend files modified: $(tr '\n' ' ' <<< "${backend_touches}")"
printf 'backend-file tripwire: 0 backend files touched vs HEAD\n'

# ---------------------------------------------------------------------------
# 4. Fresh CPU-only gate build (never Release-mutating, never backend)
# ---------------------------------------------------------------------------
cd "${REPO_ROOT}"
if [ -e "${GATE_BUILD_DIR}" ]; then
    printf 'removing stale gate build dir: %s\n' "${GATE_BUILD_DIR}"
    rm -rf -- "${GATE_BUILD_DIR}"
fi
run_step 240 "cmake configure (GPU_BACKEND=CPU)" \
    cmake -S . -B "${GATE_BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug -DGPU_BACKEND=CPU -DBUILD_TESTING=ON
run_step 420 "cmake build (CPU)" cmake --build "${GATE_BUILD_DIR}"

# ---------------------------------------------------------------------------
# 5. Post-build coupling guard: the seam binary must link the real controller
# ---------------------------------------------------------------------------
command -v nm >/dev/null 2>&1 || die "nm is unavailable; the post-build coupling guard cannot run"
for nm_target in "${NM_POSITIVE_TARGETS[@]}"; do
    [ -x "${GATE_BUILD_DIR}/${nm_target}" ] ||
        die "expected CPU test binary ${GATE_BUILD_DIR}/${nm_target} was not produced"
    NM_OUT="${WORKDIR}/nm-demangled-$(basename "${nm_target}").txt"
    timeout 60 nm -C "${GATE_BUILD_DIR}/${nm_target}" >"${NM_OUT}" 2>&1 ||
        die "nm -C failed on ${nm_target}"
    for pattern in "${NM_REQUIRED_PATTERNS[@]}"; do
        hits="$(grep -c -- "$pattern" "${NM_OUT}" || true)"
        [ "${hits}" -ge 1 ] ||
            die "coupling guard: ${nm_target} contains no '${pattern}' symbol; the seam test is not linked against the real PipelineController"
        printf 'coupling guard: %s hits for %s in %s\n' "${hits}" "${pattern}" "${nm_target}"
    done
done
# Negative control: the Qt-free cpu_smoke_harness links only azu_test_core, so
# the real controller's internal symbol must be absent there. Its presence would
# mean the controller TU leaked into the Qt-free lane (or the guard lost teeth).
[ -x "${GATE_BUILD_DIR}/${NM_NEGATIVE_TARGET}" ] ||
    die "expected CPU control binary ${GATE_BUILD_DIR}/${NM_NEGATIVE_TARGET} was not produced"
NM_NEG_OUT="${WORKDIR}/nm-demangled-control.txt"
timeout 60 nm -C "${GATE_BUILD_DIR}/${NM_NEGATIVE_TARGET}" >"${NM_NEG_OUT}" 2>&1 ||
    die "nm -C failed on ${NM_NEGATIVE_TARGET}"
neg_hits="$(grep -c -- "${NM_ABSENT_PATTERN}" "${NM_NEG_OUT}" || true)"
[ "${neg_hits}" -eq 0 ] ||
    die "negative control ${NM_NEGATIVE_TARGET} unexpectedly contains '${NM_ABSENT_PATTERN}'; Qt-free lane must not link the controller"
printf 'negative control: 0 hits for %s in %s (Qt-free lane stays controller-free)\n' "${NM_ABSENT_PATTERN}" "${NM_NEGATIVE_TARGET}"

# Post-build coupling guard (todo 29): fusion_ui_validation_contract must link the
# REAL Qt-free fusion UI model TU. The three mangled namespace symbols only exist
# if src/gui/FusionUiModel.cpp is really in azu_test_core; a source-only replica
# that re-declares its own look-alike functions carries none of them (and could
# not link alongside the real definitions anyway).
UI_MODEL_TARGET="tests/fusion_ui_validation_contract"
[ -x "${GATE_BUILD_DIR}/${UI_MODEL_TARGET}" ] ||
    die "expected CPU test binary ${GATE_BUILD_DIR}/${UI_MODEL_TARGET} was not produced"
NM_UI_OUT="${WORKDIR}/nm-demangled-fusion_ui.txt"
timeout 60 nm -C "${GATE_BUILD_DIR}/${UI_MODEL_TARGET}" >"${NM_UI_OUT}" 2>&1 ||
    die "nm -C failed on ${UI_MODEL_TARGET}"
for pattern in "kfusion::gui::validateFusionHyperparams" \
               "kfusion::gui::applyFusionPreset" \
               "kfusion::gui::computeOverlapDisplay"; do
    ui_hits="$(grep -c -- "$pattern" "${NM_UI_OUT}" || true)"
    [ "${ui_hits}" -ge 1 ] ||
        die "coupling guard: ${UI_MODEL_TARGET} contains no '${pattern}' symbol; the UI test is not linked against the real src/gui/FusionUiModel.cpp"
    printf 'coupling guard: %s hits for %s in %s\n' "${ui_hits}" "${pattern}" "${UI_MODEL_TARGET}"
done

# ---------------------------------------------------------------------------
# 6. Required CPU tests only (label cpu), then prove the label actually ran
# ---------------------------------------------------------------------------
CTEST_LIST="${WORKDIR}/ctest-list.txt"
printf '\n$ [timeout 60s] ctest --test-dir %s -L %s -N\n' "${GATE_BUILD_DIR}" "${CPU_LABEL}"
timeout 60 ctest --test-dir "${GATE_BUILD_DIR}" -L "${CPU_LABEL}" -N >"${CTEST_LIST}" 2>&1 ||
    die "ctest -N failed to enumerate the ${CPU_LABEL} lane"
cat "${CTEST_LIST}"
for t in "${REQUIRED_TESTS[@]}"; do
    grep -Eq "Test +#[0-9]+: +${t}[[:space:]]*$" "${CTEST_LIST}" ||
        die "required CPU test '${t}' is not registered in the ${CPU_LABEL} lane"
done
run_step 420 "ctest run (-L ${CPU_LABEL})" \
    ctest --test-dir "${GATE_BUILD_DIR}" -L "${CPU_LABEL}" --output-on-failure

printf '\n=== cpu gate summary ===\n'
printf 'required_cpu_lane: pass\ncpu_tests: %s\n' "${REQUIRED_TESTS[*]}"
emit_deferral_matrix
printf '\nCPU GATE PASS\n'
