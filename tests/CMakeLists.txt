# CPU-only headless test harness (big-fix todo 2).
# Reached ONLY from the root CMakeLists when GPU_BACKEND=CPU and BUILD_TESTING
# are both set, so this directory never configures CUDA/HIP, never invokes
# nvcc/hipcc, and its targets never carry CUDA_ENABLED/HIP_ENABLED (the root
# scopes those macros to the KinectFusionQt target).

# The root sets CMAKE_AUTOMOC/AUTOUIC/AUTORCC ON for Qt GUI targets. The
# controller is not a QObject and the harness is display-free: keep moc/uic/rcc
# off for everything here, including the Qt6::Core-linked controller library.
set(CMAKE_AUTOMOC OFF)
set(CMAKE_AUTOUIC OFF)
set(CMAKE_AUTORCC OFF)

# Qt6::Core is required ONLY by azu_test_pipeline (QMetaObject::invokeMethod /
# qApp in PipelineController.cpp). No Widgets, no OpenGL, no display.
find_package(Qt6 6.2 REQUIRED COMPONENTS Core)

set(AZU_TEST_DEFS AZU_PIPELINE_TEST_SEAM)

# ---- azu_test_core: Qt-free numeric/export/CPU contract library ----
# Mirrors the production CPU translation units (sensor data, preprocessing,
# tracking, TSDF, meshing, export writers, Logger/Timer) minus app/GUI code.
# Qt is not linked here; if any of these TUs ever gains a Qt dependency the
# link step will prove it.
file(GLOB AZU_TEST_CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/sensor/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/tracking/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/tsdf/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/meshing/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/export/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/utils/*.cpp"
)

add_library(azu_test_core STATIC ${AZU_TEST_CORE_SOURCES}
    # big-fix todo 27: the camera is pure Eigen math with no GL/display/device
    # dependency, unlike PreviewRenderer/ShaderProgram, so it is listed as an
    # explicit source instead of a src/rendering/*.cpp glob. The camera contract
    # below must exercise THIS translation unit, not a local replica.
    "${CMAKE_SOURCE_DIR}/src/rendering/Camera.cpp"
    # big-fix todo 29: the fusion UI validation / preset-staging / metrics-band
    # model is Qt-free (FusionHyperparams + stdlib only, no Widgets/OpenGL/device),
    # unlike ControlPanel/MetricsPanel, so it is an explicit source rather than a
    # src/gui/*.cpp glob. The UI contract below must exercise THIS real model TU --
    # the same code the widgets compile through the src/gui glob -- not a replica.
    "${CMAKE_SOURCE_DIR}/src/gui/FusionUiModel.cpp"
    # GUI GL-launch diagnostics: the EGL-code decode, desktop-only format ladder
    # and requested-vs-actual comparison are stdlib-only (no Qt/OpenGL/EGL/display),
    # unlike OpenGLWidget/PreviewRenderer, so it is an explicit source rather than a
    # src/gui/*.cpp glob. gl_diagnostics_contract below must exercise THIS real TU,
    # which is exactly why the module keeps every QSurfaceFormat touch in the
    # separate Qt-only header gui/GlFormatQt.h instead of here.
    "${CMAKE_SOURCE_DIR}/src/gui/GlDiagnostics.cpp"
)
target_compile_definitions(azu_test_core PUBLIC ${AZU_TEST_DEFS})
target_include_directories(azu_test_core PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src"
    "${TINYGLTF_DIR}"
    ${EIGEN3_INCLUDE_DIR}
    ${FREENECT_INCLUDE_DIRS}
)
if(Eigen3_FOUND)
    target_link_libraries(azu_test_core PUBLIC Eigen3::Eigen)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(azu_test_core PUBLIC OpenMP::OpenMP_CXX)
endif()
target_link_libraries(azu_test_core PUBLIC Threads::Threads)
if(FREENECT_FOUND)
    # Same build-level truth as production: HAVE_FREENECT is defined at the
    # directory level when pkg-config found libfreenect; absent libfreenect
    # compiles KinectSensor.cpp on its HAVE_FREENECT=0 no-device path.
    target_link_libraries(azu_test_core PUBLIC ${FREENECT_LIBRARIES})
endif()
set_target_properties(azu_test_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The harness Debug build type must still clear the seam's hard 5 s logical
# budget: the production per-frame CPU work (preprocess + TSDF integrate +
# 640x480 raycast) measured 4-5 s at -O0 and only ~0.5 s at -O2. This is
# harness build tuning; the asserted contracts are unchanged.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(azu_test_core PRIVATE -O2)
endif()

# ---- azu_test_pipeline: real controller translation unit, Qt6::Core-linked ----
# Compiles the REAL src/app/PipelineController.cpp (the mechanical gate below
# makes its removal a configure-time FATAL_ERROR). Sensor/tracking/TSDF/meshing/
# export/util dependencies arrive through azu_test_core; Qt6::Core provides
# QMetaObject/qApp. No display, no device, no GPU runtime.
add_library(azu_test_pipeline STATIC
    "${CMAKE_SOURCE_DIR}/src/app/PipelineController.cpp"
    "${CMAKE_SOURCE_DIR}/src/app/FrameTrace.cpp"
)
target_compile_definitions(azu_test_pipeline PUBLIC ${AZU_TEST_DEFS})
target_link_libraries(azu_test_pipeline PUBLIC azu_test_core Qt6::Core)
set_target_properties(azu_test_pipeline PROPERTIES
    AUTOMOC OFF AUTOUIC OFF AUTORCC OFF
    POSITION_INDEPENDENT_CODE ON
)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(azu_test_pipeline PRIVATE -O2)
endif()

# ---- Test executables ----
add_executable(cpu_smoke_harness cpu_smoke_harness.cpp)
target_compile_definitions(cpu_smoke_harness PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(cpu_smoke_harness PRIVATE azu_test_core)

add_executable(pipeline_test_seam_smoke pipeline_test_seam_smoke.cpp)
target_compile_definitions(pipeline_test_seam_smoke PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(pipeline_test_seam_smoke PRIVATE azu_test_pipeline)

add_test(NAME cpu_smoke_harness COMMAND cpu_smoke_harness)
set_tests_properties(cpu_smoke_harness PROPERTIES LABELS cpu TIMEOUT 120)

add_test(NAME pipeline_test_seam_smoke COMMAND pipeline_test_seam_smoke)
set_tests_properties(pipeline_test_seam_smoke PROPERTIES
    LABELS cpu TIMEOUT 120
    # Logical budget inside the binary is 16 frames / 5 s; TIMEOUT caps the
    # process even if a worker thread misbehaves.
)

# ---- big-fix todo 9: CPU-only pipeline stop/start contract ----
# Repeated start -> (optionally inject one frame) -> stop cycles against the
# real controller. Empty-queue and non-empty-queue shutdown paths are both
# exercised; a lost shutdown notification deadlocks stop() and the TIMEOUT
# (never an in-test sleep) is what fails the run. Links azu_test_pipeline so
# the real PipelineController is what actually runs (gate 4 below enforces it).
add_executable(pipeline_stop_contract pipeline_stop_contract.cpp)
target_compile_definitions(pipeline_stop_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(pipeline_stop_contract PRIVATE azu_test_pipeline)

add_test(NAME pipeline_stop_contract COMMAND pipeline_stop_contract)
set_tests_properties(pipeline_stop_contract PROPERTIES
    LABELS cpu TIMEOUT 120
    # 50 cycles; TIMEOUT caps the process if stop() ever regresses to a
    # lost-wakeup deadlock. Measured total wall time is a small fraction of
    # this (see .omo/evidence/big-fix/pipeline-stop-loop.txt).
)

# ---- big-fix todo 24: CPU-only hyperparameter / callback contract ----
# Drives the real controller while it is RUNNING: per-frame depth-band owner
# snapshots (tracking + integration), a live setHyperparams() reaching the
# preprocessor, setNumThreads() parked to a frame boundary, and the UI
# subscriber observed running with callback_mutex_ released. Every wait is
# bounded by an injected-frame count and a generation counter the production
# loops advance, so TIMEOUT (never an in-test sleep) is the hang detector.
# Links azu_test_pipeline so the real PipelineController runs (gate 6 below).
add_executable(pipeline_hyperparams_contract pipeline_hyperparams_contract.cpp)
target_compile_definitions(pipeline_hyperparams_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(pipeline_hyperparams_contract PRIVATE azu_test_pipeline)

add_test(NAME pipeline_hyperparams_contract COMMAND pipeline_hyperparams_contract)
set_tests_properties(pipeline_hyperparams_contract PROPERTIES
    LABELS cpu TIMEOUT 120
    # ~48 injected frames at 500ms spacing plus bounded waits; the cap protects
    # against a worker that stops draining its queue entirely.
)

# ---- big-fix todo 25: CPU-only reset/start-state + queue backpressure contract ----
# Drives the real controller through start/stop/reset cycles with the tracking and
# integration workers parked at seam gates (a parked worker holds no lock, so the
# queues fill behind it and the retain-latest eviction policy becomes exactly
# determined). Asserts a cleared motion model after start(), counted raw and
# integration evictions, and metrics_mutex_ free at shutdown. Every wait is
# decided by a production-advanced counter; TIMEOUT is the hang detector.
add_executable(pipeline_state_contract pipeline_state_contract.cpp)
target_compile_definitions(pipeline_state_contract PRIVATE ${AZU_PIPELINE_TEST_DEFS})
target_link_libraries(pipeline_state_contract PRIVATE azu_test_pipeline)

add_test(NAME pipeline_state_contract COMMAND pipeline_state_contract)
set_tests_properties(pipeline_state_contract PROPERTIES
    LABELS cpu TIMEOUT 120
    # ~6 bounded pump phases; the cap protects against a parked worker that is
    # never released or a queue that stops draining.
)

# ---- big-fix todo 25: CPU-only mesh request / cadence contract ----
# Pins the mesh cadence through the seam so exactly one automatic (bootstrap)
# request exists per start, then issues explicit versioned requests, including
# one issued while an extraction is parked mid-worker and one pending across
# reset(). Asserts per-version extraction/publish tagging, the stale-generation
# drop, the reset drain, and that tracking advances while a mesh runs.
add_executable(pipeline_mesh_cadence_contract pipeline_mesh_cadence_contract.cpp)
target_compile_definitions(pipeline_mesh_cadence_contract PRIVATE ${AZU_PIPELINE_TEST_DEFS})
target_link_libraries(pipeline_mesh_cadence_contract PRIVATE azu_test_pipeline)

add_test(NAME pipeline_mesh_cadence_contract COMMAND pipeline_mesh_cadence_contract)
set_tests_properties(pipeline_mesh_cadence_contract PROPERTIES
    LABELS cpu TIMEOUT 120
    # Six real CPU marching-cubes extractions plus bounded pumps; the cap is the
    # hang detector for a mesh worker that never wakes.
)

# ---- big-fix todo 26: CPU-only shared export helper (exportMesh) contract ----
# Drives the REAL controller (never started, so no worker thread and no
# extraction can race a write) through exportMesh(path, writer_fn) and proves:
# an empty shared mesh refuses without ever invoking the writer; a published
# mesh reaches writer_fn as the same object with the exact path; false and
# throwing writers are failed exports; and exportPLY/exportGLB are byte-for-
# byte identical to direct PLYExporter/GLBExporter calls (both route through
# the one helper). The only wait is the documented bounded 5 s mesh-request
# timeout on the empty-mesh path; no device, display, GPU or thread timing.
add_executable(pipeline_export_mesh_contract pipeline_export_mesh_contract.cpp)
target_compile_definitions(pipeline_export_mesh_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(pipeline_export_mesh_contract PRIVATE azu_test_pipeline)

add_test(NAME pipeline_export_mesh_contract COMMAND pipeline_export_mesh_contract)
set_tests_properties(pipeline_export_mesh_contract PROPERTIES
    LABELS cpu TIMEOUT 120
    # 5 s request timeout + four small file writes; the cap protects against a
    # request wait that never becomes bounded.
)

# ---- big-fix todo 6: Marching Cubes fail-before-fix contracts ----
# marching_cubes_table_contract is CPU-only (header + integer math, no volume)
# and validates all 256 rows against an independent crossing-edge oracle: it
# fails on ANY crossing-edge mismatch. marching_cubes_sphere_contract drives the
# real MarchingCubes::extract over an analytic sphere, but its Gate 0 rejects an
# unrepaired edge_table[213/214/215] BEFORE extraction (fail-closed: extracting
# those configs would read uninitialized edge data), so the fixture can never
# reach UB through the table. Both contracts were RED at todo 6 against the
# corrupt rows; todo 7 repaired them to 0x83f/0xb35/0xa3c in the shared header,
# so Gate 0 now passes and the extraction + geometry assertions run (GREEN).
add_executable(marching_cubes_table_contract marching_cubes_table_contract.cpp)
target_compile_definitions(marching_cubes_table_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(marching_cubes_table_contract PRIVATE azu_test_core)

add_executable(marching_cubes_sphere_contract marching_cubes_sphere_contract.cpp)
target_compile_definitions(marching_cubes_sphere_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(marching_cubes_sphere_contract PRIVATE azu_test_core)

add_test(NAME marching_cubes_table_contract COMMAND marching_cubes_table_contract)
set_tests_properties(marching_cubes_table_contract PROPERTIES LABELS cpu TIMEOUT 120)

add_test(NAME marching_cubes_sphere_contract COMMAND marching_cubes_sphere_contract)
set_tests_properties(marching_cubes_sphere_contract PROPERTIES LABELS cpu TIMEOUT 120)

# ---- big-fix todo 8: CPU TSDF empty-sentinel + stale-host-state contract ----
# Public TSDFVolume API only, small 8^3 volume, every voxel scanned (never
# sampled): no device, display, GPU, sensor, thread timing or filesystem. RED at
# todo 8 before the fix (reset() filled tsdf = 0.0f, i.e. the zero isosurface,
# and setParams() cleared host voxels only on a resolution change), GREEN after.
add_executable(tsdf_reset_contract tsdf_reset_contract.cpp)
target_compile_definitions(tsdf_reset_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(tsdf_reset_contract PRIVATE azu_test_core)

add_test(NAME tsdf_reset_contract COMMAND tsdf_reset_contract)
set_tests_properties(tsdf_reset_contract PROPERTIES LABELS cpu TIMEOUT 120)

# ---- big-fix todo 10: CPU-only mesh validation + export reject contract ----
# MeshData::validate() invariants and the three public export entry points
# (PLYExporter::writeBinary / writeASCII, GLBExporter::write). Every defect case
# must be rejected BEFORE the exporter writes: the contract asserts the target
# path stays absent, so an export that opens the file first cannot pass. Valid
# meshes must still produce a PLY/GLB whose format magic (and, for GLB, the
# header total length) matches what was written. Files live only in a
# process-unique subdirectory of std::filesystem::temp_directory_path(), removed
# at exit; nothing is written inside the repository and there is no GPU, sensor,
# display, thread-timing or network access.
add_executable(mesh_validation_contract mesh_validation_contract.cpp)
target_compile_definitions(mesh_validation_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(mesh_validation_contract PRIVATE azu_test_core)

add_test(NAME mesh_validation_contract COMMAND mesh_validation_contract)
set_tests_properties(mesh_validation_contract PROPERTIES LABELS cpu TIMEOUT 120)

# ---- big-fix todo 11: CPU coordinate floor-semantics contract ----
# Shared kfusion::utils::floorToInt() (include/utils/CoordinateMath.h) plus the
# two CPU production surfaces that consume it: TSDFVolume::worldToVoxel/getTSDF
# and the ICP model-pixel projection. Public API only: a small negative-origin,
# power-of-two-voxel volume is driven through worldToVoxel/getTSDF(fallback)/
# voxelToWorld, raycast runs the getTSDF path end to end, and ICPTracker::track
# is exercised with finite, NaN and Inf live vertices through its public entry
# point (ICPResult::valid_live_points is the witness for the finite guard). No
# device, display, GPU, sensor, thread timing or filesystem. RED at todo 11
# before the product fix: worldToVoxel truncated with static_cast<int>, so a
# below-origin point aliased to voxel (0,0,0) and the NaN/Inf live vertices were
# counted as valid live; GREEN once both paths floor and reject non-finite input.
add_executable(coordinate_rounding_contract coordinate_rounding_contract.cpp)
target_compile_definitions(coordinate_rounding_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(coordinate_rounding_contract PRIVATE azu_test_core)

add_test(NAME coordinate_rounding_contract COMMAND coordinate_rounding_contract)
set_tests_properties(coordinate_rounding_contract PROPERTIES LABELS cpu TIMEOUT 120)

# ---- big-fix todo 12: CPU ICP Huber weighting + finite-rejection contract ----
# Public ICPTracker::track + buildFramePyramid seams only. The fixture is 10
# inliers (err 0.01, w 1) + 1 outlier (err 0.08, w 0.25) at identity poses with
# exactly one level-0 iteration, so t_z decouples and locks the canonical Huber
# triple: A += w*JJt, b -= J*(w*e), objective = sum psi(|e|). It discriminates
# against unweighted curvature (-0.12/11.1), w^2 curvature + w^2 e
# (-0.105/10.1625) and the (w*e)^2 objective (0.0014/11). A +Inf model normal
# and a NaN model depth (NaN residual with all six Jacobian entries finite)
# must be rejected so the 11 valid correspondences reproduce the clean result
# bit-for-bit; the pyramid section locks the 2x downsample depth-jump guard
# (nearest sample kept across a > max(0.03, 0.05*d_min) span, smoothing and
# sentinel handling unchanged). Single-threaded via setNumThreads(1); every
# scenario runs twice and compares bit-for-bit. No device, display, GPU,
# sensor, thread timing or filesystem.
add_executable(icp_weighting_contract icp_weighting_contract.cpp)
target_compile_definitions(icp_weighting_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(icp_weighting_contract PRIVATE azu_test_core)

add_test(NAME icp_weighting_contract COMMAND icp_weighting_contract)
set_tests_properties(icp_weighting_contract PROPERTIES LABELS cpu TIMEOUT 120)

# ---- big-fix todo 13: CPU ICP numeric-policy contract ----
# Public ICPTracker::track / params() / setParams() only. One single-threaded
# contract locks the canonical CPU numeric policy: adaptive Tikhonov damping with
# ill-conditioned escalation, translation and rotation step caps, squared-norm
# model/live-normal gates, determinant-corrected SO(3) projection, the acceptable
# final-step success rule, angle-threshold clamping, and an identity default
# ICPResult::pose. Every expected number is an independent hand-derived double
# oracle (closed-form normal-equation solutions), not a product-code snapshot.
# No device, display, GPU, sensor, thread timing or filesystem.
add_executable(icp_numeric_policy_contract icp_numeric_policy_contract.cpp)
target_compile_definitions(icp_numeric_policy_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(icp_numeric_policy_contract PRIVATE azu_test_core)

add_test(NAME icp_numeric_policy_contract COMMAND icp_numeric_policy_contract)
set_tests_properties(icp_numeric_policy_contract PROPERTIES LABELS cpu TIMEOUT 120)

# ---- big-fix todo 14: CPU TSDF integration determinism contract ----
# Public TSDFVolume + MarchingCubes API only. Locks the two-phase deterministic
# merge: three repeated runs produce byte-identical serialized voxels AND identical
# MarchingCubes mesh bytes, and OpenMP thread counts 1/2/4 each reproduce the
# single-thread canonical (y, x, march-step) fold. A hand-derived double oracle folds
# one two-march-step voxel by hand (weight/tsdf/color), so a lost update or dropped
# step changes it — never compared against product output. No device, display, GPU,
# sensor, thread timing or filesystem. RED against the pre-fix racy integrateCPU
# (threads 2/4 diverge from threads 1); GREEN once the stateful merge is serial.
# big-fix-two T0.9: fused plane sits on the true surface (no half-voxel bias),
# weight counts frames, volume identical across thread counts.
add_executable(fusion_bias_contract fusion_bias_contract.cpp)
target_compile_definitions(fusion_bias_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(fusion_bias_contract PRIVATE azu_test_core)
add_test(NAME fusion_bias_contract COMMAND fusion_bias_contract)
set_tests_properties(fusion_bias_contract PROPERTIES LABELS cpu TIMEOUT 120)

# big-fix-two T0.10: raycast hits lie on the fused surface; no phantom
# surfaces from behind or at oblique views.
add_executable(raycast_accuracy_contract raycast_accuracy_contract.cpp)
target_compile_definitions(raycast_accuracy_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(raycast_accuracy_contract PRIVATE azu_test_core)
add_test(NAME raycast_accuracy_contract COMMAND raycast_accuracy_contract)
set_tests_properties(raycast_accuracy_contract PROPERTIES LABELS cpu TIMEOUT 120)

# big-fix-two T0.11: ICP projects into the model image with the pose the model
# was raycast at (ModelFrame::pose), not the latest tracked pose.
add_executable(icp_refpose_contract icp_refpose_contract.cpp)
target_compile_definitions(icp_refpose_contract PRIVATE ${AZU_TEST_DEFS})
target_include_directories(icp_refpose_contract PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(icp_refpose_contract PRIVATE azu_test_core)
add_test(NAME icp_refpose_contract COMMAND icp_refpose_contract)
set_tests_properties(icp_refpose_contract PROPERTIES LABELS cpu TIMEOUT 120)

# big-fix-two T0.3: capture fps decays to 0 and the stall is flagged when
# frames stop (the hang used to look like a healthy 30 fps).
add_executable(pipeline_stall_visibility_contract pipeline_stall_visibility_contract.cpp)
target_compile_definitions(pipeline_stall_visibility_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(pipeline_stall_visibility_contract PRIVATE azu_test_pipeline)
add_test(NAME pipeline_stall_visibility_contract COMMAND pipeline_stall_visibility_contract)
set_tests_properties(pipeline_stall_visibility_contract PROPERTIES LABELS cpu TIMEOUT 60)

# big-fix-two T0.12: closed-loop synthetic trajectory through the real
# controller (preprocess, ICP, tracking policy, fusion, raycast).
add_executable(pipeline_trajectory_contract pipeline_trajectory_contract.cpp)
target_compile_definitions(pipeline_trajectory_contract PRIVATE ${AZU_TEST_DEFS})
target_include_directories(pipeline_trajectory_contract PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pipeline_trajectory_contract PRIVATE azu_test_pipeline)
add_test(NAME pipeline_trajectory_contract COMMAND pipeline_trajectory_contract)
set_tests_properties(pipeline_trajectory_contract PROPERTIES LABELS cpu TIMEOUT 180)

add_executable(tracking_policy_contract tracking_policy_contract.cpp)
target_compile_definitions(tracking_policy_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(tracking_policy_contract PRIVATE azu_test_core)
add_test(NAME tracking_policy_contract COMMAND tracking_policy_contract)
set_tests_properties(tracking_policy_contract PROPERTIES LABELS cpu TIMEOUT 60)

# big-fix-two T0.13: loss -> relocalization -> recovery, with a live preview
# while lost.
add_executable(pipeline_relocalization_contract pipeline_relocalization_contract.cpp)
target_compile_definitions(pipeline_relocalization_contract PRIVATE ${AZU_TEST_DEFS})
target_include_directories(pipeline_relocalization_contract PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pipeline_relocalization_contract PRIVATE azu_test_pipeline)
add_test(NAME pipeline_relocalization_contract COMMAND pipeline_relocalization_contract)
set_tests_properties(pipeline_relocalization_contract PROPERTIES LABELS cpu TIMEOUT 180)

# big-fix-two T0.4: end-to-end smoke through the REAL sensor path, with
# libfakenect preloaded replaying a dump generated from the synthetic room.
add_executable(make_fake_dump make_fake_dump.cpp)
target_include_directories(make_fake_dump PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(make_fake_dump PRIVATE azu_test_core)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Sphere-tracing 60 frames at -O0 exceeds the fixture timeout; same -O2
    # harness tuning as azu_test_core/azu_test_pipeline above.
    target_compile_options(make_fake_dump PRIVATE -O2)
endif()
add_executable(smoke_fakenect smoke_fakenect.cpp)
target_link_libraries(smoke_fakenect PRIVATE azu_test_pipeline)
find_library(AZU_FAKENECT_LIB NAMES fakenect
    PATHS ${FREENECT_LIBRARY_DIRS} /usr/local/lib /usr/local/lib64 /usr/lib64 /usr/lib
    PATH_SUFFIXES fakenect NO_DEFAULT_PATH)
if(FREENECT_FOUND AND AZU_FAKENECT_LIB)
    set(AZU_FAKE_DUMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/fake_dump")
    add_test(NAME make_fake_dump COMMAND make_fake_dump ${AZU_FAKE_DUMP_DIR} 60)
    set_tests_properties(make_fake_dump PROPERTIES LABELS cpu FIXTURES_SETUP fake_dump TIMEOUT 60)
    add_test(NAME smoke_fakenect COMMAND smoke_fakenect ${CMAKE_CURRENT_BINARY_DIR}/smoke_fakenect.ply)
    set_tests_properties(smoke_fakenect PROPERTIES
        LABELS cpu FIXTURES_REQUIRED fake_dump TIMEOUT 60 SKIP_RETURN_CODE 77
        # Throughput thresholds: never share the CPU with other OpenMP tests.
        RUN_SERIAL TRUE
        ENVIRONMENT "LD_PRELOAD=${AZU_FAKENECT_LIB};FAKENECT_PATH=${AZU_FAKE_DUMP_DIR}")
    message(STATUS "smoke_fakenect: using ${AZU_FAKENECT_LIB}")
else()
    message(STATUS "smoke_fakenect: libfakenect not found, test not registered")
endif()

# Rotation: a full 360 degree turn in place in the synthetic room, lockstep,
# through the real controller.
add_executable(pipeline_spin_contract pipeline_spin_contract.cpp)
target_compile_definitions(pipeline_spin_contract PRIVATE ${AZU_TEST_DEFS})
target_include_directories(pipeline_spin_contract PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pipeline_spin_contract PRIVATE azu_test_pipeline)
add_test(NAME pipeline_spin_contract COMMAND pipeline_spin_contract)
set_tests_properties(pipeline_spin_contract PROPERTIES LABELS cpu TIMEOUT 300)
# The same turn with the GUI's Room preset (volume centred on the camera).
add_test(NAME pipeline_spin_room_preset COMMAND pipeline_spin_contract)
set_tests_properties(pipeline_spin_room_preset PROPERTIES
    LABELS cpu TIMEOUT 300 ENVIRONMENT "AZU_SPIN_PRESET=room")

add_executable(tsdf_integration_race_contract tsdf_integration_race_contract.cpp)
target_compile_definitions(tsdf_integration_race_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(tsdf_integration_race_contract PRIVATE azu_test_core)

add_test(NAME tsdf_integration_race_contract COMMAND tsdf_integration_race_contract)
set_tests_properties(tsdf_integration_race_contract PROPERTIES LABELS cpu TIMEOUT 120)

# ---- big-fix todo 15: CPU TSDF raycast + configured depth-band contracts ----
# Public TSDFVolume / FusionHyperparams API only. tsdf_raycast_contract locks the
# raycast march against hand-derived crossings on a synthetic lattice (Z-depth band
# derivation for off-axis rays, entry AND exit crossings, non-finite samples resolving
# to the canonical empty state, vertex/normal/color written in lockstep).
# tsdf_integration_min_depth_contract locks the CONFIGURED near/far depth gates through
# integrate() (a hand-derived single-candidate fold moves when min_depth moves; the
# historical hard-coded 0.1f floor makes two volumes identical), the non-finite pixel
# gate, TSDFParams' documented defaults and the single-owner sync helper.
# tsdf_subvoxel_thin_feature_contract sweeps sub-voxel plate phases: representable
# phases resolve at the piecewise-linear reconstruction zero, unrepresentable phases
# report nothing, and the configured band truncates both ends of the march.
# No device, display, GPU, sensor, thread timing or filesystem.
add_executable(tsdf_raycast_contract tsdf_raycast_contract.cpp)
target_compile_definitions(tsdf_raycast_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(tsdf_raycast_contract PRIVATE azu_test_core)

add_executable(tsdf_integration_min_depth_contract tsdf_integration_min_depth_contract.cpp)
target_compile_definitions(tsdf_integration_min_depth_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(tsdf_integration_min_depth_contract PRIVATE azu_test_core)

add_executable(tsdf_subvoxel_thin_feature_contract tsdf_subvoxel_thin_feature_contract.cpp)
target_compile_definitions(tsdf_subvoxel_thin_feature_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(tsdf_subvoxel_thin_feature_contract PRIVATE azu_test_core)

add_test(NAME tsdf_raycast_contract COMMAND tsdf_raycast_contract)
set_tests_properties(tsdf_raycast_contract PROPERTIES LABELS cpu TIMEOUT 120)

add_test(NAME tsdf_integration_min_depth_contract COMMAND tsdf_integration_min_depth_contract)
set_tests_properties(tsdf_integration_min_depth_contract PROPERTIES LABELS cpu TIMEOUT 120)

add_test(NAME tsdf_subvoxel_thin_feature_contract COMMAND tsdf_subvoxel_thin_feature_contract)
set_tests_properties(tsdf_subvoxel_thin_feature_contract PROPERTIES LABELS cpu TIMEOUT 120)

# ---- big-fix todo 16: CPU Marching Cubes observed-frontier contracts ----
# Public TSDFVolume + MarchingCubes::extract only, on direct synthetic volumes.
# Locks the canonical frontier rules: an unobserved or non-finite voxel samples
# as +1.0f whatever tsdf literal it carries; a cube is evaluated from its
# crossing edges so an unrelated unobserved corner cannot delete it; a crossing
# edge is emitted only with two supported endpoints; a normal that is not usable
# yields no vertex instead of a fabricated +Z; the volume border takes a
# one-sided gradient instead of an invented out-of-bounds sample; repeated
# extraction is byte-identical. Expectations are independent oracles (integer
# edge algebra over the shared table, hand-derived positions, directed-edge
# topology), never product snapshots. No device, display, GPU, sensor, thread
# timing or filesystem.
add_executable(marching_cubes_frontier_contract marching_cubes_frontier_contract.cpp)
target_compile_definitions(marching_cubes_frontier_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(marching_cubes_frontier_contract PRIVATE azu_test_core)

add_test(NAME marching_cubes_frontier_contract COMMAND marching_cubes_frontier_contract)
set_tests_properties(marching_cubes_frontier_contract PROPERTIES LABELS cpu TIMEOUT 120)

# ---- big-fix todo 17: CPU exact edge-key welding + mesh invariant contracts ----
# Public TSDFVolume + MarchingCubes::extract only, direct synthetic volumes. The
# welding contract proves the CPU weld maps each physical crossing edge to exactly
# ONE global vertex by canonical edge identity (lower voxel + axis) rather than a
# quantized world position, and that the welded vertex count equals an independent
# distinct-edge-key oracle. The color contract locks exact color/geometry lockstep,
# hasColors()/validate(), and that a vertex color is interpolated with the SAME
# parameter that produced its position. The truncation/progress contract drives the
# CPU triangle budget (full-triangle boundary, MeshData::truncated, no partial row)
# and the ProgressCallback thread-safety (never concurrent, monotonic, final 1.0,
# deterministic sequence, graceful tiny-resolution handling). Expectations are
# independent oracles, never product snapshots. No device, display, GPU, sensor,
# thread timing or filesystem.
add_executable(mesh_welding_contract mesh_welding_contract.cpp)
target_compile_definitions(mesh_welding_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(mesh_welding_contract PRIVATE azu_test_core)

add_executable(mesh_color_invariant_contract mesh_color_invariant_contract.cpp)
target_compile_definitions(mesh_color_invariant_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(mesh_color_invariant_contract PRIVATE azu_test_core)

add_executable(mesh_truncation_progress_contract mesh_truncation_progress_contract.cpp)
target_compile_definitions(mesh_truncation_progress_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(mesh_truncation_progress_contract PRIVATE azu_test_core)

add_test(NAME mesh_welding_contract COMMAND mesh_welding_contract)
set_tests_properties(mesh_welding_contract PROPERTIES LABELS cpu TIMEOUT 120)

add_test(NAME mesh_color_invariant_contract COMMAND mesh_color_invariant_contract)
set_tests_properties(mesh_color_invariant_contract PROPERTIES LABELS cpu TIMEOUT 120)

add_test(NAME mesh_truncation_progress_contract COMMAND mesh_truncation_progress_contract)
set_tests_properties(mesh_truncation_progress_contract PROPERTIES LABELS cpu TIMEOUT 120)

# ---- big-fix todo 18: CPU winding-convention derivation contract ----
# Public TSDFVolume + MarchingCubes::extract only. Derives the authoritative CPU
# triangle-winding convention from signed-volume / outward-normal mathematics over
# an analytic signed-distance sphere (negative inside, positive outside, verified
# voxel-by-voxel before extraction) — NEVER from a hardcoded expected tri_table
# order and never from current product output. Section 0 self-derives the rule
# V_signed = (1/6) sum a.(b x c) > 0 <=> outward on a hand-built unit cube; the
# sphere sections then require V_signed > 0 (inscribed band 0.99*V0 < V < V0),
# closed-manifold winding, a strict per-face radial test, a per-cube both-candidate
# derivation over independently interpolated crossing vertices, and byte-identical
# geometry across repeats and OpenMP threads 1/2/4. No device, display, GPU,
# sensor, thread timing or filesystem. LOCK at todo 18 (the reverse emission order
# already satisfies the derived rule; the todo proves it and pins it, with the
# mutation flip of the emission order as the falsification witness).
add_executable(marching_cubes_winding_contract marching_cubes_winding_contract.cpp)
target_compile_definitions(marching_cubes_winding_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(marching_cubes_winding_contract PRIVATE azu_test_core)

add_test(NAME marching_cubes_winding_contract COMMAND marching_cubes_winding_contract)
set_tests_properties(marching_cubes_winding_contract PROPERTIES LABELS cpu TIMEOUT 120)

# ---- big-fix todo 19: CPU color convergence + linear GLB color contracts ----
# color_convergence_contract is public CPU API only (TSDFVolume integrate/raycast/
# extractGlobalPointCloud + the shared utils::ColorMath policy): it locks the uint8
# sRGB -> float sRGB widening as exactly b/255 over all 256 bytes, the one shared
# float -> byte policy as round-to-nearest with an exact 256-byte round trip - it
# clamps every finite value into [0,1] before scaling and rejects only NaN/Inf, leaving
# the output byte untouched - fusion that
# converges to the true mean instead of stalling in the old per-update uint8 rounding
# dead zone, EMPTY_COLOR as the exact 128/255 quotient publishing byte 128, and that
# every extraction boundary emits bytes the shared policy reproduces. No device,
# display, GPU, sensor, thread timing or filesystem.
# glb_linear_color_contract exports REAL temporary .glb files through GLBExporter and
# reads them back with tinygltf (declarations only; the tinygltf definitions come from
# GLBExporter.cpp.o inside azu_test_core), so COLOR_0 is verified as written on disk:
# componentType FLOAT, type VEC4, count == POSITION count, alpha exactly 1.0, and RGB
# within 1e-5 of an independent double sRGB EOTF oracle written in the test - including
# the piecewise knee (byte 10 linear branch, byte 11 power branch) and a positive
# witness that the value is NOT the raw byte/255 the old exporter emitted. Refusal
# paths write no file at all, exports are byte-identical across repeats, and the PLY
# path keeps the exact uint8 sRGB triples. Scratch files live under
# std::filesystem::temp_directory_path() and are removed.
add_executable(color_convergence_contract color_convergence_contract.cpp)
target_compile_definitions(color_convergence_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(color_convergence_contract PRIVATE azu_test_core)

add_executable(glb_linear_color_contract glb_linear_color_contract.cpp)
target_compile_definitions(glb_linear_color_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(glb_linear_color_contract PRIVATE azu_test_core)

# big-fix todo 20: CPU raw-depth validity + out-of-band meters rejection + hole-fill
# sentinel semantics. depth_domain_contract walks all 65536 raw codes against an
# independent double oracle (predicate raw==0||raw>=2047, the reciprocal pole at
# raw ~1084.6, band acceptance == raws 1..954 for [0.30,2.50], rejection is exactly
# +0.0f, the encoder never clamps to a wall, hole fill only touches raw 0 and leaves
# 2047 alone). depth_ema_determinism_contract pins the temporal EMA to a source
# snapshot: an asymmetric 500/548/586 fixture whose hand-derived answers are raw 534
# (wall) and raw 575 (centre), with raw 586 asserted absent as the in-place
# contamination signature, plus digest identity across repeats and 1/2/4 threads.
add_executable(depth_domain_contract depth_domain_contract.cpp)
target_compile_definitions(depth_domain_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(depth_domain_contract PRIVATE azu_test_core)

add_executable(depth_ema_determinism_contract depth_ema_determinism_contract.cpp)
target_compile_definitions(depth_ema_determinism_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(depth_ema_determinism_contract PRIVATE azu_test_core)

add_test(NAME depth_domain_contract COMMAND depth_domain_contract)
set_tests_properties(depth_domain_contract PROPERTIES LABELS cpu TIMEOUT 120)

add_test(NAME depth_ema_determinism_contract COMMAND depth_ema_determinism_contract)
set_tests_properties(depth_ema_determinism_contract PROPERTIES LABELS cpu TIMEOUT 120)

add_test(NAME color_convergence_contract COMMAND color_convergence_contract)
set_tests_properties(color_convergence_contract PROPERTIES LABELS cpu TIMEOUT 120)

# ---- big-fix todo 21: CPU CAS border mode + guidance luma contract ----
# cas_border_contract is CPU-only: the shared border helper
# (include/sensor/BorderMode.h), the public sr::applyEASU_CPU product pass and the
# AZU_PIPELINE_TEST_SEAM guidance accessors. Section A pins the reflection rule
# over extents 1..1024 and proves the +-1 stencil is border-BLIND (reflect and
# clamp agree at -1 and at extent), so only the 4x4 Catmull-Rom resample, whose
# taps reach -2 and extent+1, can tell the two modes apart. Section B compares
# REAL applyEASU_CPU output against an independent double oracle per border mode
# and requires every divergent destination byte (mode gap >= 3, the floor that the
# 1-byte float-vs-double oracle gap allows) to sit on the reflect side. Sections
# C/D/E drive the real buildSuperResolutionGuidance() over seven synthetic 640x480
# patterns against an independent double CAS + luma oracle, pin
# (0.299R+0.587G+0.114B)/255 over the POST-CAS bytes, prove the derived [0,1]
# range (bit-exact 0.0f/1.0f on flat black/white), exclude R/B-swap, Rec.709,
# missing-/255 and unsharpened-source expectations, and lock the luma digest
# across repeats and OpenMP threads 1/2/4. No device, display, GPU, sensor, thread
# timing or filesystem.
add_executable(cas_border_contract cas_border_contract.cpp)
target_compile_definitions(cas_border_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(cas_border_contract PRIVATE azu_test_core)

add_test(NAME cas_border_contract COMMAND cas_border_contract)
set_tests_properties(cas_border_contract PROPERTIES LABELS cpu TIMEOUT 120)

# ---- big-fix todo 22: CPU upscaled-RGB availability + non-finite sharpness ----
# sr_upscaled_contract is public CPU API only: the SignalConditioner /
# Preprocessor availability contract, the real process() production path, and the
# public sr::applyEASU_CPU / sr::applyCAS_CPU passes (never a reimplementation of
# EASU or CAS). It locks that a fresh or reset conditioner reports the upscaled
# buffer UNAVAILABLE even though getSrRgbUpscaled() already holds
# 640x480x3 zero bytes (the black-texture hazard), that a valid CPU frame
# publishes exactly FRAME_W*scale * FRAME_H*scale * 3 bytes for scale 2/3/4
# byte-identically to the product passes over that same frame, that a frame
# rejected on size retires the PREVIOUS frame's availability too, that scale 1
# and every scale outside [2,4] fail closed (scale 1 used to publish an
# original-resolution image whose byte count a 2x-sized consumer would read out
# of bounds), that reset() clears id and availability even while the buffer keeps
# a valid-looking 2x size, and that the CUDA/HIP-labelled preprocessor reports
# the buffer unavailable without any GPU code being constructed, compiled or
# run. Sharpness: NaN, +Inf and -Inf in the real CPU CAS pass must be
# byte-identical to the finite 0.0f output (std::clamp passed NaN straight
# through to static_cast<uint8_t>(NaN * 255.0f), which is UB and on x86
# blackened the frame), with the 0.5f/1.0f runs as the non-vacuity witness.
# Deterministic across repeats and OpenMP threads 1/2/4. No device, display, GPU,
# sensor, thread timing or filesystem.
add_executable(sr_upscaled_contract sr_upscaled_contract.cpp)
target_compile_definitions(sr_upscaled_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(sr_upscaled_contract PRIVATE azu_test_core)

add_test(NAME sr_upscaled_contract COMMAND sr_upscaled_contract)
set_tests_properties(sr_upscaled_contract PROPERTIES LABELS cpu TIMEOUT 120)

# ---- big-fix todo 23: CPU depth/RGB pairing staleness + latest-frame semantics ----
# kinect_pairing_contract drives a REAL kfusion::sensor::KinectSensor through the
# AZU_PIPELINE_TEST_SEAM injectors only: default-constructed, so there is no
# device, no libfreenect call, no capture thread, no display, no GPU and no
# sleep. It locks the 50 ms sync window (strict `<`, so exactly 50 ms is stale),
# the stale-side-drop / newer-side-retain recovery on both ledgers, the fact that
# a rejected pair consumes no frame id, that the product callback observes
# sync_mutex_ as free, that a published frame carries the bytes and millisecond
# timestamps of the samples it actually paired with, and that the ready slot holds
# ONE frame: three accepted pairs leave id 8 and recycle ids 6/7, and the second
# getLatestFrame() returns nullptr instead of an older queued frame. Expected
# values are the fixture's own injected payloads, never product snapshots.
add_executable(kinect_pairing_contract kinect_pairing_contract.cpp)
target_compile_definitions(kinect_pairing_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(kinect_pairing_contract PRIVATE azu_test_core)

add_test(NAME kinect_pairing_contract COMMAND kinect_pairing_contract)
set_tests_properties(kinect_pairing_contract PROPERTIES LABELS cpu TIMEOUT 120)

# Regression for the c311c69 hang: real-device timestamp trace plus synthetic
# 60 MHz streams across a full depth/RGB phase-slip cycle and a counter wrap.
add_executable(kinect_pairing_realtrace_contract kinect_pairing_realtrace_contract.cpp)
target_compile_definitions(kinect_pairing_realtrace_contract PRIVATE ${AZU_TEST_DEFS}
    AZU_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(kinect_pairing_realtrace_contract PRIVATE azu_test_core)
add_test(NAME kinect_pairing_realtrace_contract COMMAND kinect_pairing_realtrace_contract)
set_tests_properties(kinect_pairing_realtrace_contract PROPERTIES LABELS cpu TIMEOUT 120)

add_test(NAME glb_linear_color_contract COMMAND glb_linear_color_contract)
set_tests_properties(glb_linear_color_contract PROPERTIES LABELS cpu TIMEOUT 120)

# ---- big-fix todo 27: CPU camera orbit/pan/basis + first-frame tick clamp ----
# camera_basis_contract drives the REAL src/rendering/Camera.cpp (linked through
# azu_test_core, gate 10 below) through its public API only: cameraBasis(),
# orbitEye(), viewMatrix(), pan(), move(), rotate(), zoom() and the pure
# clampFrameDeltaSeconds() tick policy. Expected matrices are hand-derived from
# the orbit-eye definition eye = target - distance*forward plus orthonormal and
# on-axis-target invariants, so the pre-fix "+distance*sin(pitch)" eye and the
# pre-fix roll-free pan basis both fail instead of silently passing. Locks the
# negative-Y convention at four yaw octants, roll-aware right/up for pan and
# WASD/QE, Orbit<->Free view continuity, and that only the first tick is
# replaced by the nominal one. Pure Eigen arithmetic: no OpenGL context, device,
# display, GPU, sensor, network, wall-clock oracle, thread timing or filesystem.
add_executable(camera_basis_contract camera_basis_contract.cpp)
target_compile_definitions(camera_basis_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(camera_basis_contract PRIVATE azu_test_core)

add_test(NAME camera_basis_contract COMMAND camera_basis_contract)
set_tests_properties(camera_basis_contract PROPERTIES LABELS cpu TIMEOUT 120)

# ---- big-fix todo 29: CPU Qt-free fusion UI validation / band model ----
# fusion_ui_validation_contract drives the REAL src/gui/FusionUiModel.cpp (linked
# through azu_test_core, gate 11 below) through its public namespace only. It
# locks the cross-field hyperparameter rules the Qt spin boxes cannot enforce on
# their own (min<max strictly, the 4.0 m device depth ceiling, the 2..8x
# truncation/voxel ratio, finiteness / positive geometry), that presets stage ONE
# atomic complete value rather than mutating fields piecemeal, that
# FusionHyperparams::defaults() is the one valid seed, that icp_valid_model == -1
# maps to the textual "--" UNKNOWN band (never a premature warming colour), and
# that the metrics band cache re-polishes only on a band TRANSITION. No Qt
# Widgets, OpenGL context, display, device, GPU, sensor, thread timing or
# filesystem. Expectations are independent rule statements, never product
# snapshots.
add_executable(fusion_ui_validation_contract fusion_ui_validation_contract.cpp)
target_compile_definitions(fusion_ui_validation_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(fusion_ui_validation_contract PRIVATE azu_test_core)

add_test(NAME fusion_ui_validation_contract COMMAND fusion_ui_validation_contract)
set_tests_properties(fusion_ui_validation_contract PROPERTIES LABELS cpu TIMEOUT 120)

# ---- GUI GL-launch diagnostics: CPU Qt-free EGL/ladder/floor contract ----
# gl_diagnostics_contract drives the REAL src/gui/GlDiagnostics.cpp (linked through
# azu_test_core, gate 16 below) through its public namespace only. It locks that the
# canonical EGL codes 0x3000..0x300E decode to their exact names (incl. the reported
# 0x3009 BAD_MATCH and the 0x3003/0x3005/0x3008/0x300D/0x300E family) while unknown
# codes fall back to EGL_UNKNOWN(0x..); that BAD_MATCH's hint names BOTH real-world
# causes and the __EGL_VENDOR_LIBRARY_FILENAMES dual-vendor lever while SUCCESS and
# unknown codes carry no hint; that decodeEglErrorFromText pulls the bare "3009" out
# of Qt's free-form text and ignores code-free messages; that the format ladder is
# desktop-only, opens at 3.3 Core + MSAA, trades MSAA away as its one downgrade and
# is then EXHAUSTED (no sub-3.3 / no ES rung); that meetsRendererFloor accepts only
# DesktopGL+Core+>=3.3; and that compareGlFormat treats an MSAA downgrade as
# adequate-but-not-matching while an ES/sub-3.3 context is inadequate. No Qt
# Widgets, OpenGL context, EGL, display, device, GPU, sensor or filesystem.
# Expectations are independent rule statements, never product snapshots.
add_executable(gl_diagnostics_contract gl_diagnostics_contract.cpp)
target_compile_definitions(gl_diagnostics_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(gl_diagnostics_contract PRIVATE azu_test_core)

add_test(NAME gl_diagnostics_contract COMMAND gl_diagnostics_contract)
set_tests_properties(gl_diagnostics_contract PROPERTIES LABELS cpu TIMEOUT 120)

# ---- big-fix todo 30: CPU PLY writer schema / byte / failure contract ----
# ply_writer_contract drives the REAL src/export/PLYExporter.cpp (carried by
# azu_test_core through its src/export/*.cpp glob; gate 12 below) through BOTH
# public entry points and PARSES the bytes on disk. It locks: byte-exact binary
# output (magic, binary_little_endian, header counts, per-record stride, total
# length, host-independent little-endian float/int bit patterns); one shared
# element/property schema across binary and ASCII (only the "format" token may
# differ) with ASCII values re-parsing to the same mesh (uint8 sRGB stays raw);
# byte-identical repeats; and every failure creating no file -- empty / out-of-
# range / non-multiple-of-3 / color-size-mismatch refused before open, an
# unwritable target refused at open, and a REAL post-open RLIMIT_FSIZE write
# fault removing the partial file. Files live in a process-unique temp dir. No
# device, display, GPU, sensor, thread timing or network.
add_executable(ply_writer_contract ply_writer_contract.cpp)
target_compile_definitions(ply_writer_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(ply_writer_contract PRIVATE azu_test_core)

add_test(NAME ply_writer_contract COMMAND ply_writer_contract)
set_tests_properties(ply_writer_contract PROPERTIES LABELS cpu TIMEOUT 120)

# ---- big-fix todo 31: CPU GLB writer glTF-validity / linear-color contract ----
# glb_writer_contract drives the REAL src/export/GLBExporter.cpp (carried by
# azu_test_core through its src/export/*.cpp glob; gate 13 below) through its public
# write() entry point and PARSES THE GLB CONTAINER ITSELF - magic, version,
# totalLength, 4-aligned space-padded JSON chunk and zero-padded BIN chunk - then
# parses the JSON chunk with the bundled nlohmann JSON, so glTF validity is asserted
# against the spec numbers on disk (5126/5125, VEC3/VEC4/SCALAR, POSITION min+max,
# 4-aligned byteOffsets, 34962/34963 targets, no zero-count accessor, disjoint
# in-buffer views) before any tinygltf round-trip is trusted. It locks the
# right-handed Y-up mapping (x,-y,-z), unit normals with the documented fallback for
# a zero-length input, winding preserved through the position weld, COLOR_0 as LINEAR
# float RGB against an independent double EOTF oracle, min/max that describe the data
# even beyond the old +-1e9 sentinel, refusal of every invalid mesh with no file, a
# real post-open RLIMIT_FSIZE fault leaving no partial GLB with its real cause
# reported, and byte-identical repeats. Files live in a process-unique temp dir.
add_executable(glb_writer_contract glb_writer_contract.cpp)
target_compile_definitions(glb_writer_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(glb_writer_contract PRIVATE azu_test_core)

add_test(NAME glb_writer_contract COMMAND glb_writer_contract)
set_tests_properties(glb_writer_contract PROPERTIES LABELS cpu TIMEOUT 120)

# ---- big-fix todo 33: CPU logger/timer hygiene + stats denominator contracts ----
# logger_timer_contract drives the REAL src/utils/Logger.cpp and src/utils/Timer.cpp
# (gate 14 below demands both TUs inside azu_test_core, so a local replica cannot
# link). Expectations are protocol properties, never snapshots: threshold semantics
# for the typed and variadic paths, the exact record grammar including the fixed
# three-digit millisecond field and the 10-column tag, stamp validity plus
# monotonicity across a burst, and a single-write integrity test in which foreign
# threads stream multi-insert lines to the same unbuffered std::cerr while four
# logger threads emit 1600 uniquely-tokened records - every logger record must still
# arrive whole, exactly once, because one composed string is one write(2). It also
# proves ScopedTimer performs ZERO allocations per below-threshold scope using a
# global operator new counter and a name longer than any SSO buffer, and that the
# const char* name contract is compile-enforced. stderr is captured to a private
# temp file through dup2 and removed.
add_executable(logger_timer_contract logger_timer_contract.cpp)
target_compile_definitions(logger_timer_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(logger_timer_contract PRIVATE azu_test_core)

# stats_denominator_contract drives the REAL TSDFVolume / SignalConditioner /
# FrameData diagnostic channels (gate 15 below). It locks that every published mean
# divides by its CONTRIBUTING SAMPLE COUNT, through oracles that need no product
# snapshot: max/count <= mean <= max over a one-valid-pixel TSDF frame (a raster
# denominator misses by four orders of magnitude), mean == max exactly on uniform
# EMA and uniform-depth fixtures, the two FrameData counters partitioning the
# raster, depth_filtered counting the fixture's own invalid pixels, padding
# invariance across the invalid sentinel versus an out-of-band distance, a
# zero-contribution frame publishing exactly 0.0 and never NaN, and every channel
# creating NO file when its env gate is absent. Each scenario needs a fresh env and
# a fresh working directory, so it runs in its own fork+execve child inside a
# private mkdtemp tree the parent inspects and removes.
add_executable(stats_denominator_contract stats_denominator_contract.cpp)
target_compile_definitions(stats_denominator_contract PRIVATE ${AZU_TEST_DEFS})
target_link_libraries(stats_denominator_contract PRIVATE azu_test_core)

add_test(NAME logger_timer_contract COMMAND logger_timer_contract)
set_tests_properties(logger_timer_contract PROPERTIES LABELS cpu TIMEOUT 120)

add_test(NAME stats_denominator_contract COMMAND stats_denominator_contract)
set_tests_properties(stats_denominator_contract PROPERTIES LABELS cpu TIMEOUT 120)

# ---- Mechanical anti-synthetic gate (configure-time, comment-hardened) ----
# big-fix todo 3 hardening of the todo 2 gate:
#   F1 fix: a raw text MATCHES was satisfiable by a printf-only replica that
#       named the seam identifiers only inside a comment while still linking
#       azu_test_pipeline. Comments are now stripped and the gate demands a
#       CALL SHAPE (receiver '.'/'>'/':' + name + '('), so prose no longer
#       configures. scripts/test-cpu-big-fix.sh adds the matching post-build
#       nm -C coupling guard; a source-only replica with its own look-alike
#       methods dies there instead.
#   F2 fix: the controller TU check used to anchor the joined SOURCES string,
#       i.e. it false-failed as soon as a second TU was appended. The real TU
#       is now required as one path element anywhere in the list.

# Strip /* ... */ then // comments from a source-text variable. Stripping only
# ever removes text, so an over-eager match can fail the gate closed but can
# never manufacture a call shape.
function(azu_strip_c_comments out_var in_var)
    set(_azu_t "${${in_var}}")
    string(REGEX REPLACE "/\\*([^*]|\\*+[^*/])*\\*+/" "" _azu_t "${_azu_t}")
    string(REGEX REPLACE "//[^\n]*" "" _azu_t "${_azu_t}")
    set(${out_var} "${_azu_t}" PARENT_SCOPE)
endfunction()

# 1) The registered pipeline_test_seam_smoke source must actually CALL the
#    seam API (a synthetic queue/worker replica fails here).
get_target_property(AZU_SEAM_TEST_SOURCES pipeline_test_seam_smoke SOURCES)
if(NOT AZU_SEAM_TEST_SOURCES)
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: pipeline_test_seam_smoke has no SOURCES.")
endif()
set(AZU_SEAM_TEST_TEXT "")
foreach(_azu_seam_src ${AZU_SEAM_TEST_SOURCES})
    get_filename_component(_azu_seam_abs "${_azu_seam_src}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT EXISTS "${_azu_seam_abs}")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: registered pipeline test source ${_azu_seam_abs} does not exist.")
    endif()
    file(READ "${_azu_seam_abs}" _azu_seam_txt)
    azu_strip_c_comments(_azu_seam_txt_code _azu_seam_txt)
    string(APPEND AZU_SEAM_TEST_TEXT "${_azu_seam_txt_code}")
endforeach()
foreach(_azu_required_call startWithoutSensorForTests injectRawFrameForTests)
    if(NOT AZU_SEAM_TEST_TEXT MATCHES "[.>:]${_azu_required_call}[ \t\r\n]*\\(")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: the pipeline_test_seam_smoke source must call ${_azu_required_call}() "
            "on the real PipelineController (a comment or string literal naming it is not a call). "
            "A replacement queue/worker replica does not satisfy this todo.")
    endif()
endforeach()

# 2) azu_test_pipeline must compile the REAL controller translation unit.
#    Required as a source-list ELEMENT (any position), not as the string tail.
get_target_property(AZU_PIPELINE_LIB_SOURCES azu_test_pipeline SOURCES)
if(NOT AZU_PIPELINE_LIB_SOURCES)
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: azu_test_pipeline has no SOURCES.")
endif()
set(AZU_PIPELINE_LIB_HAS_REAL_TU FALSE)
foreach(_azu_pipeline_src ${AZU_PIPELINE_LIB_SOURCES})
    file(TO_CMAKE_PATH "${_azu_pipeline_src}" _azu_pipeline_norm)
    if(_azu_pipeline_norm MATCHES "(^|/)src/app/PipelineController\\.cpp$")
        set(AZU_PIPELINE_LIB_HAS_REAL_TU TRUE)
    endif()
endforeach()
if(NOT AZU_PIPELINE_LIB_HAS_REAL_TU)
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: azu_test_pipeline SOURCES must include src/app/PipelineController.cpp "
        "as one source element (any position), got: ${AZU_PIPELINE_LIB_SOURCES}")
endif()
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/src/app/PipelineController.cpp")
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: src/app/PipelineController.cpp is missing from the source tree.")
endif()

# 3) The registered pipeline test executable must link azu_test_pipeline,
#    and azu_test_pipeline must link Qt6::Core (QMetaObject/qApp contract).
get_target_property(AZU_SEAM_TEST_LINKS pipeline_test_seam_smoke LINK_LIBRARIES)
if(NOT "${AZU_SEAM_TEST_LINKS}" MATCHES "azu_test_pipeline")
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: pipeline_test_seam_smoke must link azu_test_pipeline, "
        "got: ${AZU_SEAM_TEST_LINKS}")
endif()
get_target_property(AZU_PIPELINE_LIB_LINKS azu_test_pipeline LINK_LIBRARIES)
if(NOT "${AZU_PIPELINE_LIB_LINKS}" MATCHES "Qt6::Core")
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: azu_test_pipeline must link Qt6::Core "
        "(PipelineController.cpp uses QMetaObject::invokeMethod/qApp), got: ${AZU_PIPELINE_LIB_LINKS}")
endif()

# 4) big-fix todo 9: pipeline_stop_contract must be the REAL controller seam
#    start/stop test, not a synthetic queue/worker replica. Same hardened
#    discipline as gate 1: comments are stripped, so only a genuine call shape
#    (receiver '.'/'>'/'::' + name + '(') on the stripped source configures, and
#    the target must link azu_test_pipeline.
get_target_property(AZU_STOP_TEST_SOURCES pipeline_stop_contract SOURCES)
if(NOT AZU_STOP_TEST_SOURCES)
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: pipeline_stop_contract has no SOURCES.")
endif()
set(AZU_STOP_TEST_TEXT "")
foreach(_azu_stop_src ${AZU_STOP_TEST_SOURCES})
    get_filename_component(_azu_stop_abs "${_azu_stop_src}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT EXISTS "${_azu_stop_abs}")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: registered pipeline_stop_contract source ${_azu_stop_abs} does not exist.")
    endif()
    file(READ "${_azu_stop_abs}" _azu_stop_txt)
    azu_strip_c_comments(_azu_stop_txt_code _azu_stop_txt)
    string(APPEND AZU_STOP_TEST_TEXT "${_azu_stop_txt_code}")
endforeach()
foreach(_azu_stop_call startWithoutSensorForTests stop injectRawFrameForTests)
    if(NOT AZU_STOP_TEST_TEXT MATCHES "[.>:]${_azu_stop_call}[ \t\r\n]*\\(")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: the pipeline_stop_contract source must call ${_azu_stop_call}() "
            "on the real PipelineController (a comment or string literal naming it is not a call). "
            "A replacement queue/worker replica does not satisfy this todo.")
    endif()
endforeach()
get_target_property(AZU_STOP_TEST_LINKS pipeline_stop_contract LINK_LIBRARIES)
if(NOT "${AZU_STOP_TEST_LINKS}" MATCHES "azu_test_pipeline")
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: pipeline_stop_contract must link azu_test_pipeline, "
        "got: ${AZU_STOP_TEST_LINKS}")
endif()

# 5) big-fix todo 23: kinect_pairing_contract must drive the REAL KinectSensor
#    pairing path. Same discipline as gate 1/4: comments are stripped first, so
#    only a code shape configures the gate.
get_target_property(AZU_KINECT_TEST_SOURCES kinect_pairing_contract SOURCES)
if(NOT AZU_KINECT_TEST_SOURCES)
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: kinect_pairing_contract has no SOURCES.")
endif()
set(AZU_KINECT_TEST_TEXT "")
foreach(_azu_kp_src ${AZU_KINECT_TEST_SOURCES})
    get_filename_component(_azu_kp_abs "${_azu_kp_src}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT EXISTS "${_azu_kp_abs}")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: registered kinect_pairing_contract source ${_azu_kp_abs} does not exist.")
    endif()
    file(READ "${_azu_kp_abs}" _azu_kp_txt)
    azu_strip_c_comments(_azu_kp_txt_code _azu_kp_txt)
    string(APPEND AZU_KINECT_TEST_TEXT "${_azu_kp_txt_code}")
endforeach()

# The fixture must be the product sensor object. A default-constructed member
# declaration has no call parentheses, so the accepted shapes are a call to
# KinectSensor() OR a member/object declared with the fully qualified product
# type name. The qualification is what carries the weight: a look-alike replica
# would have to re-declare kfusion::sensor::KinectSensor, which collides with the
# azu_test_core header the target is required to link (gate below).
set(_azu_kp_type_ok FALSE)
if(AZU_KINECT_TEST_TEXT MATCHES "[.>:]KinectSensor[ \t\r\n]*\\(")
    set(_azu_kp_type_ok TRUE)
endif()
if(AZU_KINECT_TEST_TEXT MATCHES "kfusion::sensor::[ \t\r\n]*KinectSensor[ \t\r\n]+[A-Za-z_][A-Za-z0-9_]*[ \t\r\n]*;")
    set(_azu_kp_type_ok TRUE)
endif()
if(NOT _azu_kp_type_ok)
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: the kinect_pairing_contract source must construct the real "
        "kfusion::sensor::KinectSensor (call shape, or a fully qualified product-type declaration). "
        "A local queue/pairing replica does not satisfy this todo.")
endif()
foreach(_azu_kp_call injectDepthForTests injectRgbForTests syncMutexIsFreeForTests
                     readyFrameCountForTests freeFrameCountForTests getLatestFrame)
    if(NOT AZU_KINECT_TEST_TEXT MATCHES "[.>:]${_azu_kp_call}[ \t\r\n]*\\(")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: the kinect_pairing_contract source must call ${_azu_kp_call}() "
            "on the real KinectSensor (a comment or string literal naming it is not a call). "
            "A replacement pairing/queue replica does not satisfy this todo.")
    endif()
endforeach()
get_target_property(AZU_KINECT_TEST_LINKS kinect_pairing_contract LINK_LIBRARIES)
if(NOT "${AZU_KINECT_TEST_LINKS}" MATCHES "azu_test_core")
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: kinect_pairing_contract must link azu_test_core (the real sensor TU), "
        "got: ${AZU_KINECT_TEST_LINKS}")
endif()

# 6) big-fix todo 24: pipeline_hyperparams_contract must drive the REAL
#    controller hyperparameter/callback path, not a synthetic replica. Same
#    discipline as gate 1/4/5: comments are stripped, so only a genuine call
#    shape (receiver '.'/'>'/'::' + name + '(') configures the gate, and the
#    target must link azu_test_pipeline. The required calls are exactly the
#    propagation and serialization surfaces this todo owns: seam start/injection,
#    the hyperparameter setter and reader, both per-frame band observations, and
#    the thread-count/SR-scale observations.
get_target_property(AZU_HYPER_TEST_SOURCES pipeline_hyperparams_contract SOURCES)
if(NOT AZU_HYPER_TEST_SOURCES)
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: pipeline_hyperparams_contract has no SOURCES.")
endif()
set(AZU_HYPER_TEST_TEXT "")
foreach(_azu_hyper_src ${AZU_HYPER_TEST_SOURCES})
    get_filename_component(_azu_hyper_abs "${_azu_hyper_src}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT EXISTS "${_azu_hyper_abs}")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: registered pipeline_hyperparams_contract source ${_azu_hyper_abs} does not exist.")
    endif()
    file(READ "${_azu_hyper_abs}" _azu_hyper_txt)
    azu_strip_c_comments(_azu_hyper_txt_code _azu_hyper_txt)
    string(APPEND AZU_HYPER_TEST_TEXT "${_azu_hyper_txt_code}")
endforeach()
foreach(_azu_hyper_call startWithoutSensorForTests injectRawFrameForTests
                        setHyperparams hyperparamsSnapshot setNumThreads
                        lastTrackingDepthBandForTests lastIntegrationDepthBandForTests
                        appliedSrScaleForTests appliedThreadCountForTests
                        callbackMutexIsFreeForTests setFrameReadyCallback stop)
    if(NOT AZU_HYPER_TEST_TEXT MATCHES "[.>:]${_azu_hyper_call}[ \t\r\n]*\\(")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: the pipeline_hyperparams_contract source must call ${_azu_hyper_call}() "
            "on the real PipelineController (a comment or string literal naming it is not a call). "
            "A queue/parameter replica does not satisfy this todo.")
    endif()
endforeach()
get_target_property(AZU_HYPER_TEST_LINKS pipeline_hyperparams_contract LINK_LIBRARIES)
if(NOT "${AZU_HYPER_TEST_LINKS}" MATCHES "azu_test_pipeline")
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: pipeline_hyperparams_contract must link azu_test_pipeline, "
        "got: ${AZU_HYPER_TEST_LINKS}")
endif()

# 7) big-fix todo 25: pipeline_state_contract must drive the REAL controller
#    reset/start path and BOTH product queues. Same discipline as gate 1/4/5/6:
#    comments are stripped, so only a genuine call shape (receiver '.'/'>'/'::' +
#    name + '(') satisfies the gate. The required calls are exactly the surfaces
#    this todo owns: the motion model (currentPose / lastPoseForTests /
#    lastMotionModelForTests), both bounded queues' eviction counters as observed
#    through metricsSnapshot, the seam gates that make eviction determined, and
#    the shutdown lock probe.
get_target_property(AZU_STATE_TEST_SOURCES pipeline_state_contract SOURCES)
if(NOT AZU_STATE_TEST_SOURCES)
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: pipeline_state_contract has no SOURCES.")
endif()
set(AZU_STATE_TEST_TEXT "")
foreach(_azu_state_src ${AZU_STATE_TEST_SOURCES})
    get_filename_component(_azu_state_abs "${_azu_state_src}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT EXISTS "${_azu_state_abs}")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: registered pipeline_state_contract source ${_azu_state_abs} does not exist.")
    endif()
    file(READ "${_azu_state_abs}" _azu_state_txt)
    azu_strip_c_comments(_azu_state_txt_code _azu_state_txt)
    string(APPEND AZU_STATE_TEST_TEXT "${_azu_state_txt_code}")
endforeach()
foreach(_azu_state_call startWithoutSensorForTests injectRawFrameForTests stop reset
                        currentPose lastPoseForTests lastMotionModelForTests metricsSnapshot
                        pauseTrackingWorkerForTests pauseIntegrationWorkerForTests
                        resumePipelineWorkersForTests trackingWorkerParkedForTests
                        integrationWorkerParkedForTests rawQueueDepthForTests
                        integrationQueueDepthForTests metricsMutexIsFreeForTests
                        lastTrackingDepthBandForTests)
    if(NOT AZU_STATE_TEST_TEXT MATCHES "[.>:]${_azu_state_call}[ \t\r\n]*\\(")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: the pipeline_state_contract source must call ${_azu_state_call}() "
            "on the real PipelineController (a comment or string literal naming it is not a call). "
            "A queue/state replica does not satisfy this todo.")
    endif()
endforeach()
get_target_property(AZU_STATE_TEST_LINKS pipeline_state_contract LINK_LIBRARIES)
if(NOT "${AZU_STATE_TEST_LINKS}" MATCHES "azu_test_pipeline")
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: pipeline_state_contract must link azu_test_pipeline, "
        "got: ${AZU_STATE_TEST_LINKS}")
endif()

# 8) big-fix todo 25: pipeline_mesh_cadence_contract must drive the REAL mesh
#    worker. The required calls are the versioned request protocol (explicit
#    request + reset invalidation), the cadence setter that makes the automatic
#    request count exact, the claimed-version list the parked worker reports, and
#    the version/generation counters that carry the oracle.
get_target_property(AZU_MESH_TEST_SOURCES pipeline_mesh_cadence_contract SOURCES)
if(NOT AZU_MESH_TEST_SOURCES)
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: pipeline_mesh_cadence_contract has no SOURCES.")
endif()
set(AZU_MESH_TEST_TEXT "")
foreach(_azu_mesh_src ${AZU_MESH_TEST_SOURCES})
    get_filename_component(_azu_mesh_abs "${_azu_mesh_src}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT EXISTS "${_azu_mesh_abs}")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: registered pipeline_mesh_cadence_contract source ${_azu_mesh_abs} does not exist.")
    endif()
    file(READ "${_azu_mesh_abs}" _azu_mesh_txt)
    azu_strip_c_comments(_azu_mesh_txt_code _azu_mesh_txt)
    string(APPEND AZU_MESH_TEST_TEXT "${_azu_mesh_txt_code}")
endforeach()
foreach(_azu_mesh_call startWithoutSensorForTests injectRawFrameForTests stop reset
                       setMeshCadenceIntervalForTests requestMeshForTests
                       invalidateMeshStateForTests meshStateForTests
                       armMeshExtractionHoldForTests releaseMeshExtractionHoldForTests
                       meshHookVersionsForTests lastTrackingDepthBandForTests)
    if(NOT AZU_MESH_TEST_TEXT MATCHES "[.>:]${_azu_mesh_call}[ \t\r\n]*\\(")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: the pipeline_mesh_cadence_contract source must call ${_azu_mesh_call}() "
            "on the real PipelineController (a comment or string literal naming it is not a call). "
            "A cadence/counter replica does not satisfy this todo.")
    endif()
endforeach()
get_target_property(AZU_MESH_TEST_LINKS pipeline_mesh_cadence_contract LINK_LIBRARIES)
if(NOT "${AZU_MESH_TEST_LINKS}" MATCHES "azu_test_pipeline")
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: pipeline_mesh_cadence_contract must link azu_test_pipeline, "
        "got: ${AZU_MESH_TEST_LINKS}")
endif()

# 9) big-fix todo 26: pipeline_export_mesh_contract must drive the REAL
#    controller's shared export helper and BOTH public export wrappers. Same
#    discipline as gate 1/4/5/6/7/8: comments are stripped, so only a genuine
#    call shape (receiver '.'/'>'/'::' + name + '(') configures, and the
#    target must link azu_test_pipeline (the real controller TU).
get_target_property(AZU_EXPORT_TEST_SOURCES pipeline_export_mesh_contract SOURCES)
if(NOT AZU_EXPORT_TEST_SOURCES)
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: pipeline_export_mesh_contract has no SOURCES.")
endif()
set(AZU_EXPORT_TEST_TEXT "")
foreach(_azu_export_src ${AZU_EXPORT_TEST_SOURCES})
    get_filename_component(_azu_export_abs "${_azu_export_src}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT EXISTS "${_azu_export_abs}")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: registered pipeline_export_mesh_contract source ${_azu_export_abs} does not exist.")
    endif()
    file(READ "${_azu_export_abs}" _azu_export_txt)
    azu_strip_c_comments(_azu_export_txt_code _azu_export_txt)
    string(APPEND AZU_EXPORT_TEST_TEXT "${_azu_export_txt_code}")
endforeach()
foreach(_azu_export_call exportMesh exportPLY exportGLB sharedMesh metricsSnapshot)
    if(NOT AZU_EXPORT_TEST_TEXT MATCHES "[.>:]${_azu_export_call}[ \t\r\n]*\\(")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: the pipeline_export_mesh_contract source must call ${_azu_export_call}() "
            "on the real PipelineController (a comment or string literal naming it is not a call). "
            "An exporter-only replica does not satisfy this todo.")
    endif()
endforeach()
get_target_property(AZU_EXPORT_TEST_LINKS pipeline_export_mesh_contract LINK_LIBRARIES)
if(NOT "${AZU_EXPORT_TEST_LINKS}" MATCHES "azu_test_pipeline")
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: pipeline_export_mesh_contract must link azu_test_pipeline, "
        "got: ${AZU_EXPORT_TEST_LINKS}")
endif()

# 10) big-fix todo 27: camera_basis_contract must drive the REAL camera
#     translation unit through the real camera API. Same discipline as gate
#     1/4/5/6/7/8/9: comments are stripped, so only a genuine call shape
#     (receiver '.'/'>'/'::' + name + '(') satisfies it, and the real
#     src/rendering/Camera.cpp has to be an ELEMENT of azu_test_core's own
#     source list -- a test that re-declares its own Camera/orbitEye replica in
#     the test file cannot pass either half of this gate.
get_target_property(AZU_CAMERA_TEST_SOURCES camera_basis_contract SOURCES)
if(NOT AZU_CAMERA_TEST_SOURCES)
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: camera_basis_contract has no SOURCES.")
endif()
set(AZU_CAMERA_TEST_TEXT "")
foreach(_azu_camera_src ${AZU_CAMERA_TEST_SOURCES})
    get_filename_component(_azu_camera_abs "${_azu_camera_src}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT EXISTS "${_azu_camera_abs}")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: registered camera_basis_contract source ${_azu_camera_abs} does not exist.")
    endif()
    file(READ "${_azu_camera_abs}" _azu_camera_txt)
    azu_strip_c_comments(_azu_camera_txt_code _azu_camera_txt)
    string(APPEND AZU_CAMERA_TEST_TEXT "${_azu_camera_txt_code}")
endforeach()
# The surfaces todo 27 owns: the one orbit-eye helper, the one shared basis
# helper, the matrix both feed, the two consumers that must go through that
# basis, and the pure first-frame tick policy.
foreach(_azu_camera_call orbitEye cameraBasis viewMatrix pan move rotate zoom
                        clampFrameDeltaSeconds setMode setRoll)
    if(NOT AZU_CAMERA_TEST_TEXT MATCHES "[.>:]${_azu_camera_call}[ \t\r\n]*\\(")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: the camera_basis_contract source must call ${_azu_camera_call}() "
            "on the real kfusion::rendering::Camera / camera API "
            "(a comment or string literal naming it is not a call). "
            "A local orbit/basis replica does not satisfy this todo.")
    endif()
endforeach()
get_target_property(AZU_CAMERA_TEST_LINKS camera_basis_contract LINK_LIBRARIES)
if(NOT "${AZU_CAMERA_TEST_LINKS}" MATCHES "azu_test_core")
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: camera_basis_contract must link azu_test_core, "
        "got: ${AZU_CAMERA_TEST_LINKS}")
endif()
# Positive coupling: the library it links has to CARRY the real camera TU.
get_target_property(AZU_TEST_CORE_LIB_SOURCES azu_test_core SOURCES)
if(NOT AZU_TEST_CORE_LIB_SOURCES)
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: azu_test_core has no SOURCES.")
endif()
set(AZU_TEST_CORE_HAS_CAMERA_TU FALSE)
foreach(_azu_core_src ${AZU_TEST_CORE_LIB_SOURCES})
    file(TO_CMAKE_PATH "${_azu_core_src}" _azu_core_norm)
    if(_azu_core_norm MATCHES "(^|/)src/rendering/Camera\\.cpp$")
        set(AZU_TEST_CORE_HAS_CAMERA_TU TRUE)
    endif()
endforeach()
if(NOT AZU_TEST_CORE_HAS_CAMERA_TU)
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: azu_test_core must compile the real src/rendering/Camera.cpp "
        "as one source-list element; camera_basis_contract would otherwise test a replica "
        "instead of the product camera.")
endif()

# 11) big-fix todo 29: fusion_ui_validation_contract must drive the REAL Qt-free
#     fusion UI model translation unit, not a header-only replica the test keeps
#     to itself. Same discipline as gate 1/4/5/6/7/8/9/10: comments are stripped,
#     so only a genuine namespace-qualified CALL SHAPE (receiver '.'/'>'/'::' +
#     name + '(') satisfies the call checks, the real src/gui/FusionUiModel.cpp
#     must be an ELEMENT of azu_test_core's own source list, and the target must
#     link azu_test_core. A test that re-declares its own validateFusionHyperparams
#     replica cannot pass the source-coupling half (it would collide with the real
#     definition the linked library carries) and cannot satisfy the call shape for
#     the free functions without the :: qualification the real header demands.
get_target_property(AZU_UI_TEST_SOURCES fusion_ui_validation_contract SOURCES)
if(NOT AZU_UI_TEST_SOURCES)
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: fusion_ui_validation_contract has no SOURCES.")
endif()
set(AZU_UI_TEST_TEXT "")
foreach(_azu_ui_src ${AZU_UI_TEST_SOURCES})
    get_filename_component(_azu_ui_abs "${_azu_ui_src}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT EXISTS "${_azu_ui_abs}")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: registered fusion_ui_validation_contract source ${_azu_ui_abs} does not exist.")
    endif()
    file(READ "${_azu_ui_abs}" _azu_ui_txt)
    azu_strip_c_comments(_azu_ui_txt_code _azu_ui_txt)
    string(APPEND AZU_UI_TEST_TEXT "${_azu_ui_txt_code}")
endforeach()
# The headless surfaces todo 29 owns: the cross-field validator, the atomic preset
# stager, the overlap-band renderer, and the edge-triggered band cache's apply().
foreach(_azu_ui_call validateFusionHyperparams applyFusionPreset computeOverlapDisplay apply)
    if(NOT AZU_UI_TEST_TEXT MATCHES "[.>:]${_azu_ui_call}[ \t\r\n]*\\(")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: the fusion_ui_validation_contract source must call ${_azu_ui_call}() "
            "on the real kfusion::gui model (a comment or string literal naming it is not a call). "
            "A local validation/band replica does not satisfy this todo.")
    endif()
endforeach()
get_target_property(AZU_UI_TEST_LINKS fusion_ui_validation_contract LINK_LIBRARIES)
if(NOT "${AZU_UI_TEST_LINKS}" MATCHES "azu_test_core")
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: fusion_ui_validation_contract must link azu_test_core, "
        "got: ${AZU_UI_TEST_LINKS}")
endif()
# Positive coupling: azu_test_core has to CARRY the real fusion UI model TU.
set(AZU_TEST_CORE_HAS_UI_TU FALSE)
foreach(_azu_core_src ${AZU_TEST_CORE_LIB_SOURCES})
    file(TO_CMAKE_PATH "${_azu_core_src}" _azu_core_norm)
    if(_azu_core_norm MATCHES "(^|/)src/gui/FusionUiModel\\.cpp$")
        set(AZU_TEST_CORE_HAS_UI_TU TRUE)
    endif()
endforeach()
if(NOT AZU_TEST_CORE_HAS_UI_TU)
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: azu_test_core must compile the real src/gui/FusionUiModel.cpp "
        "as one source-list element; fusion_ui_validation_contract would otherwise test a "
        "replica instead of the product fusion UI model.")
endif()

# 12) big-fix todo 30: ply_writer_contract must drive the REAL PLY writer, not a
#     local schema/byte replica. Same discipline as gate 1/4/5/6/7/8/9/10/11:
#     comments are stripped first, so only a genuine CALL SHAPE (receiver
#     '.'/'>'/'::' + name + '(') satisfies it. Both PLY-07 halves must be exercised,
#     so writeBinary AND writeASCII call shapes are demanded, the target must link
#     azu_test_core, and azu_test_core must carry the real src/export/PLYExporter.cpp
#     (through its src/export/*.cpp glob) -- a test that re-implemented its own
#     PLY schema would link nothing and satisfy neither half.
get_target_property(AZU_PLY_TEST_SOURCES ply_writer_contract SOURCES)
if(NOT AZU_PLY_TEST_SOURCES)
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: ply_writer_contract has no SOURCES.")
endif()
set(AZU_PLY_TEST_TEXT "")
foreach(_azu_ply_src ${AZU_PLY_TEST_SOURCES})
    get_filename_component(_azu_ply_abs "${_azu_ply_src}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT EXISTS "${_azu_ply_abs}")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: registered ply_writer_contract source ${_azu_ply_abs} does not exist.")
    endif()
    file(READ "${_azu_ply_abs}" _azu_ply_txt)
    azu_strip_c_comments(_azu_ply_txt_code _azu_ply_txt)
    string(APPEND AZU_PLY_TEST_TEXT "${_azu_ply_txt_code}")
endforeach()
foreach(_azu_ply_call writeBinary writeASCII)
    if(NOT AZU_PLY_TEST_TEXT MATCHES "[.>:]${_azu_ply_call}[ \t\r\n]*\\(")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: the ply_writer_contract source must call ${_azu_ply_call}() "
            "on the real kfusion::export_io::PLYExporter (a comment or string literal naming it "
            "is not a call). A local PLY schema/byte replica does not satisfy this todo.")
    endif()
endforeach()
get_target_property(AZU_PLY_TEST_LINKS ply_writer_contract LINK_LIBRARIES)
if(NOT "${AZU_PLY_TEST_LINKS}" MATCHES "azu_test_core")
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: ply_writer_contract must link azu_test_core, "
        "got: ${AZU_PLY_TEST_LINKS}")
endif()
# Positive coupling: azu_test_core has to CARRY the real PLY writer TU.
get_target_property(AZU_PLY_CORE_SOURCES azu_test_core SOURCES)
if(NOT AZU_PLY_CORE_SOURCES)
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: azu_test_core has no SOURCES.")
endif()
set(AZU_TEST_CORE_HAS_PLY_TU FALSE)
foreach(_azu_core_src ${AZU_PLY_CORE_SOURCES})
    file(TO_CMAKE_PATH "${_azu_core_src}" _azu_core_norm)
    if(_azu_core_norm MATCHES "(^|/)src/export/PLYExporter\\.cpp$")
        set(AZU_TEST_CORE_HAS_PLY_TU TRUE)
    endif()
endforeach()
if(NOT AZU_TEST_CORE_HAS_PLY_TU)
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: azu_test_core must compile the real src/export/PLYExporter.cpp "
        "as one source-list element; ply_writer_contract would otherwise test a replica "
        "instead of the product PLY writer.")
endif()

# 13) big-fix todo 31: glb_writer_contract must drive the REAL GLB writer AND parse
#     the result two independent ways. Same discipline as gate 1/4/5/6/7/8/9/10/11/12:
#     comments are stripped first, so only a genuine CALL SHAPE (receiver
#     '.'/'>'/'::' + name + '(') satisfies it. Three shapes are demanded, because the
#     todo's acceptance criterion is "GLB JSON chunk parse AND tinygltf round-trip":
#     write() (the product writer, namespace-qualified so a `:` precedes it),
#     parse() (the nlohmann JSON parse of the JSON chunk the writer produced) and
#     LoadBinaryFromFile() (the tinygltf round trip). A test that only handed a path
#     to tinygltf would miss the parse() shape; a test that re-implemented the GLB
#     container itself would link nothing and fail the coupling half, since
#     azu_test_core must carry the real src/export/GLBExporter.cpp.
get_target_property(AZU_GLB_TEST_SOURCES glb_writer_contract SOURCES)
if(NOT AZU_GLB_TEST_SOURCES)
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: glb_writer_contract has no SOURCES.")
endif()
set(AZU_GLB_TEST_TEXT "")
foreach(_azu_glb_src ${AZU_GLB_TEST_SOURCES})
    get_filename_component(_azu_glb_abs "${_azu_glb_src}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT EXISTS "${_azu_glb_abs}")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: registered glb_writer_contract source ${_azu_glb_abs} does not exist.")
    endif()
    file(READ "${_azu_glb_abs}" _azu_glb_txt)
    azu_strip_c_comments(_azu_glb_txt_code _azu_glb_txt)
    string(APPEND AZU_GLB_TEST_TEXT "${_azu_glb_txt_code}")
endforeach()
foreach(_azu_glb_call write parse LoadBinaryFromFile)
    if(NOT AZU_GLB_TEST_TEXT MATCHES "[.>:]${_azu_glb_call}[ \t\r\n]*\\(")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: the glb_writer_contract source must call ${_azu_glb_call}() on the real "
            "writer/JSON loader (a comment or string literal naming it is not a call). A test that skips "
            "the product writer or skips the JSON-chunk parse does not satisfy this todo.")
    endif()
endforeach()
get_target_property(AZU_GLB_TEST_LINKS glb_writer_contract LINK_LIBRARIES)
if(NOT "${AZU_GLB_TEST_LINKS}" MATCHES "azu_test_core")
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: glb_writer_contract must link azu_test_core, "
        "got: ${AZU_GLB_TEST_LINKS}")
endif()
# Positive coupling: azu_test_core has to CARRY the real GLB writer TU.
get_target_property(AZU_GLB_CORE_SOURCES azu_test_core SOURCES)
if(NOT AZU_GLB_CORE_SOURCES)
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: azu_test_core has no SOURCES.")
endif()
set(AZU_TEST_CORE_HAS_GLB_TU FALSE)
foreach(_azu_core_src ${AZU_GLB_CORE_SOURCES})
    file(TO_CMAKE_PATH "${_azu_core_src}" _azu_core_norm)
    if(_azu_core_norm MATCHES "(^|/)src/export/GLBExporter\\.cpp$")
        set(AZU_TEST_CORE_HAS_GLB_TU TRUE)
    endif()
endforeach()
if(NOT AZU_TEST_CORE_HAS_GLB_TU)
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: azu_test_core must compile the real src/export/GLBExporter.cpp "
        "as one source-list element; glb_writer_contract would otherwise test a replica "
        "instead of the product GLB writer.")
endif()

# 14) big-fix todo 33: logger_timer_contract must drive the REAL Logger and Timer
#     translation units. Same discipline as gate 1/4/5/.../13: comments are stripped
#     first, so only a genuine CALL SHAPE (receiver '.'/'>'/'::' + name + '(')
#     configures the gate. The singleton accessor, both threshold surfaces and the
#     variadic path are demanded, plus a qualified ScopedTimer declaration - and the
#     library it links must CARRY both real TUs, because a test that re-declared its
#     own logger/timer could otherwise never be told apart from the product.
get_target_property(AZU_LOGTEST_SOURCES logger_timer_contract SOURCES)
if(NOT AZU_LOGTEST_SOURCES)
    message(FATAL_ERROR "ANTI-SYNTHETIC GATE: logger_timer_contract has no SOURCES.")
endif()
set(AZU_LOGTEST_TEXT "")
foreach(_azu_lsrc ${AZU_LOGTEST_SOURCES})
    get_filename_component(_azu_labs "${_azu_lsrc}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT EXISTS "${_azu_labs}")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: registered logger_timer_contract source ${_azu_labs} does not exist.")
    endif()
    file(READ "${_azu_labs}" _azu_ltxt)
    azu_strip_c_comments(_azu_ltxt_code _azu_ltxt)
    string(APPEND AZU_LOGTEST_TEXT "${_azu_ltxt_code}")
endforeach()
foreach(_azu_lcall instance setLevel currentLevel logf)
    if(NOT AZU_LOGTEST_TEXT MATCHES "[.>:]${_azu_lcall}[ \t\r\n]*\\(")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: the logger_timer_contract source must call ${_azu_lcall}() on the real "
            "kfusion::utils::Logger (a comment or string literal naming it is not a call).")
    endif()
endforeach()
if(NOT AZU_LOGTEST_TEXT MATCHES "[.>:]ScopedTimer[ \t\r\n]+[A-Za-z_][A-Za-z0-9_]*[ \t\r\n]*\\(")
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: the logger_timer_contract source must construct a qualified "
        "kfusion::utils::ScopedTimer (a replica timer class does not satisfy this todo).")
endif()
get_target_property(AZU_LOGTEST_LINKS logger_timer_contract LINK_LIBRARIES)
if(NOT "${AZU_LOGTEST_LINKS}" MATCHES "azu_test_core")
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: logger_timer_contract must link azu_test_core, got: ${AZU_LOGTEST_LINKS}")
endif()
get_target_property(AZU_LOG_CORE_SOURCES azu_test_core SOURCES)
if(NOT AZU_LOG_CORE_SOURCES)
    message(FATAL_ERROR "ANTI-SYNTHETIC GATE: azu_test_core has no SOURCES.")
endif()
foreach(_azu_needed_util Logger Timer)
    set(AZU_TEST_CORE_HAS_${_azu_needed_util}_TU FALSE)
    foreach(_azu_core_src ${AZU_LOG_CORE_SOURCES})
        file(TO_CMAKE_PATH "${_azu_core_src}" _azu_core_norm)
        if(_azu_core_norm MATCHES "(^|/)src/utils/${_azu_needed_util}\\.cpp$")
            set(AZU_TEST_CORE_HAS_${_azu_needed_util}_TU TRUE)
        endif()
    endforeach()
    if(NOT AZU_TEST_CORE_HAS_${_azu_needed_util}_TU)
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: azu_test_core must compile the real src/utils/${_azu_needed_util}.cpp "
            "as one source-list element; logger_timer_contract would otherwise test a replica.")
    endif()
endforeach()

# 15) big-fix todo 33: stats_denominator_contract must drive the REAL product paths
#     that produce each CSV row - the TSDF integrate kernel, the SignalConditioner
#     EMA stage and buildFrameData - and must READ the resulting rows back. Comments
#     are stripped first, so only a genuine CALL SHAPE configures the gate; the CSV
#     read-back is demanded through the file-opening call shape too, because a test
#     that only recomputed means in memory would prove nothing about what the
#     product publishes.
get_target_property(AZU_STATSTEST_SOURCES stats_denominator_contract SOURCES)
if(NOT AZU_STATSTEST_SOURCES)
    message(FATAL_ERROR "ANTI-SYNTHETIC GATE: stats_denominator_contract has no SOURCES.")
endif()
set(AZU_STATSTEST_TEXT "")
foreach(_azu_ssrc ${AZU_STATSTEST_SOURCES})
    get_filename_component(_azu_sabs "${_azu_ssrc}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT EXISTS "${_azu_sabs}")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: registered stats_denominator_contract source ${_azu_sabs} does not exist.")
    endif()
    file(READ "${_azu_sabs}" _azu_stxt)
    azu_strip_c_comments(_azu_stxt_code _azu_stxt)
    string(APPEND AZU_STATSTEST_TEXT "${_azu_stxt_code}")
endforeach()
foreach(_azu_scall integrate applyDepthEmaForTests buildFrameData is_open)
    if(NOT AZU_STATSTEST_TEXT MATCHES "[.>:]${_azu_scall}[ \t\r\n]*\\(")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: the stats_denominator_contract source must call ${_azu_scall}() (real product "
            "path or CSV read-back; a comment or string literal naming it is not a call).")
    endif()
endforeach()
get_target_property(AZU_STATSTEST_LINKS stats_denominator_contract LINK_LIBRARIES)
if(NOT "${AZU_STATSTEST_LINKS}" MATCHES "azu_test_core")
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: stats_denominator_contract must link azu_test_core, got: ${AZU_STATSTEST_LINKS}")
endif()
foreach(_azu_needed_stat src/tsdf/TSDFVolume src/sensor/SignalConditioner_omp src/sensor/FrameData)
    set(AZU_TEST_CORE_HAS_STAT_TU FALSE)
    foreach(_azu_core_src ${AZU_LOG_CORE_SOURCES})
        file(TO_CMAKE_PATH "${_azu_core_src}" _azu_core_norm)
        if(_azu_core_norm MATCHES "(^|/)${_azu_needed_stat}\\.cpp$")
            set(AZU_TEST_CORE_HAS_STAT_TU TRUE)
        endif()
    endforeach()
    if(NOT AZU_TEST_CORE_HAS_STAT_TU)
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: azu_test_core must compile the real ${_azu_needed_stat}.cpp; "
            "stats_denominator_contract would otherwise test a replica accumulator.")
    endif()
endforeach()

# 16) GUI GL-launch diagnostics: gl_diagnostics_contract must drive the REAL Qt-free
#     GlDiagnostics translation unit, not a header-only or in-test replica. Same
#     discipline as gate 1/4/5/6/7/8/9/10/11: comments are stripped, so only a
#     genuine namespace-qualified CALL SHAPE (receiver '.'/'>'/'::' + name + '(')
#     satisfies the call checks; the real src/gui/GlDiagnostics.cpp must be an
#     ELEMENT of azu_test_core's own source list; and the target must link
#     azu_test_core. A test that re-declared its own eglErrorName / meetsRendererFloor
#     replica could not pass the source-coupling half (it would collide with the real
#     definitions the linked library carries) and could not satisfy the call shapes
#     without the kfusion::gui '::' qualification the real header demands. The nine
#     demanded calls are the whole public surface, so a partial replica is rejected.
get_target_property(AZU_GLDIAG_TEST_SOURCES gl_diagnostics_contract SOURCES)
if(NOT AZU_GLDIAG_TEST_SOURCES)
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: gl_diagnostics_contract has no SOURCES.")
endif()
set(AZU_GLDIAG_TEST_TEXT "")
foreach(_azu_gsrc ${AZU_GLDIAG_TEST_SOURCES})
    get_filename_component(_azu_gabs "${_azu_gsrc}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT EXISTS "${_azu_gabs}")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: registered gl_diagnostics_contract source ${_azu_gabs} does not exist.")
    endif()
    file(READ "${_azu_gabs}" _azu_gtxt)
    azu_strip_c_comments(_azu_gtxt_code _azu_gtxt)
    string(APPEND AZU_GLDIAG_TEST_TEXT "${_azu_gtxt_code}")
endforeach()
foreach(_azu_gcall eglErrorName eglErrorHint decodeEglError decodeEglErrorFromText
                   glFormatLadder nextFormatRung meetsRendererFloor describeGlFormat compareGlFormat)
    if(NOT AZU_GLDIAG_TEST_TEXT MATCHES "[.>:]${_azu_gcall}[ \t\r\n]*\\(")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: the gl_diagnostics_contract source must call ${_azu_gcall}() "
            "on the real kfusion::gui module (a comment or string literal naming it is not a call). "
            "A local EGL/ladder/floor replica does not satisfy this todo.")
    endif()
endforeach()
get_target_property(AZU_GLDIAG_TEST_LINKS gl_diagnostics_contract LINK_LIBRARIES)
if(NOT "${AZU_GLDIAG_TEST_LINKS}" MATCHES "azu_test_core")
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: gl_diagnostics_contract must link azu_test_core, "
        "got: ${AZU_GLDIAG_TEST_LINKS}")
endif()
# Positive coupling: azu_test_core has to CARRY the real GL diagnostics TU.
set(AZU_TEST_CORE_HAS_GLDIAG_TU FALSE)
foreach(_azu_core_src ${AZU_TEST_CORE_LIB_SOURCES})
    file(TO_CMAKE_PATH "${_azu_core_src}" _azu_core_norm)
    if(_azu_core_norm MATCHES "(^|/)src/gui/GlDiagnostics\\.cpp$")
        set(AZU_TEST_CORE_HAS_GLDIAG_TU TRUE)
    endif()
endforeach()
if(NOT AZU_TEST_CORE_HAS_GLDIAG_TU)
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: azu_test_core must compile the real src/gui/GlDiagnostics.cpp "
        "as one source-list element; gl_diagnostics_contract would otherwise test a "
        "replica instead of the product GL diagnostics.")
endif()

message(STATUS "azu big-fix CPU test harness: cpu_smoke_harness + pipeline_test_seam_smoke + pipeline_stop_contract + pipeline_hyperparams_contract + pipeline_state_contract + pipeline_mesh_cadence_contract + pipeline_export_mesh_contract + camera_basis_contract + fusion_ui_validation_contract + ply_writer_contract + glb_writer_contract + logger_timer_contract + stats_denominator_contract + gl_diagnostics_contract registered (label=cpu)")
