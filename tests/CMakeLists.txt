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

# ---- Mechanical anti-synthetic gate (configure-time, not a grep comment) ----
# 1) The registered pipeline_test_seam_smoke source must actually call the
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
    string(APPEND AZU_SEAM_TEST_TEXT "${_azu_seam_txt}")
endforeach()
foreach(_azu_required_call startWithoutSensorForTests injectRawFrameForTests)
    if(NOT AZU_SEAM_TEST_TEXT MATCHES "${_azu_required_call}")
        message(FATAL_ERROR
            "ANTI-SYNTHETIC GATE: the pipeline_test_seam_smoke source must call ${_azu_required_call}() "
            "on the real PipelineController. A replacement queue/worker replica does not satisfy this todo.")
    endif()
endforeach()

# 2) azu_test_pipeline must compile the REAL controller translation unit.
get_target_property(AZU_PIPELINE_LIB_SOURCES azu_test_pipeline SOURCES)
if(NOT "${AZU_PIPELINE_LIB_SOURCES}" MATCHES "src/app/PipelineController\\.cpp$")
    message(FATAL_ERROR
        "ANTI-SYNTHETIC GATE: azu_test_pipeline SOURCES must include src/app/PipelineController.cpp "
        "(the real controller object), got: ${AZU_PIPELINE_LIB_SOURCES}")
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
