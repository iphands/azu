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

add_library(azu_test_core STATIC ${AZU_TEST_CORE_SOURCES})
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

message(STATUS "azu big-fix CPU test harness: cpu_smoke_harness + pipeline_test_seam_smoke registered (label=cpu)")
