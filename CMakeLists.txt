cmake_minimum_required(VERSION 3.21)
project(KinectFusionQt VERSION 1.0.0 LANGUAGES CXX)

# ============================================================
# User-facing GPU back-end selection
# ============================================================
# Pass one of these at configure time:
#   cmake -DGPU_BACKEND=HIP  ..   → force HIP/ROCm (AMD)
#   cmake -DGPU_BACKEND=CUDA ..   → force CUDA (NVIDIA)
#   cmake -DGPU_BACKEND=CPU  ..   → CPU-only, no GPU
#   cmake ..                      → auto-detect (HIP preferred on ROCm systems)
set(GPU_BACKEND "AUTO" CACHE STRING "GPU back-end: AUTO | HIP | CUDA | CPU")
set_property(CACHE GPU_BACKEND PROPERTY STRINGS AUTO HIP CUDA CPU)
message(STATUS "GPU_BACKEND selection: ${GPU_BACKEND}")

# ---- General Configuration ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(NOT CMAKE_CXX_FLAGS_RELEASE)
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
    endif()
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
    if(ipo_supported AND NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
        message(STATUS "LTO/IPO enabled")
    endif()
else()
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra")
endif()

# ============================================================
# GPU back-end detection / selection
# ============================================================
include(CheckLanguage)
set(CUDA_AVAILABLE FALSE)
set(HIP_AVAILABLE  FALSE)

if(NOT GPU_BACKEND STREQUAL "CPU")

    # ---- HIP / ROCm ----
    if(GPU_BACKEND STREQUAL "HIP" OR GPU_BACKEND STREQUAL "AUTO")
        check_language(HIP)
        if(CMAKE_HIP_COMPILER)
            enable_language(HIP)
            set(HIP_AVAILABLE TRUE)
            add_compile_definitions(HIP_ENABLED __HIP_PLATFORM_AMD__)
            set(CMAKE_HIP_STANDARD 17)
            # Ryzen 5 5650U iGPU (Vega/Cezanne)
            set(CMAKE_HIP_ARCHITECTURES gfx90c)
            message(STATUS "HIP found: ${CMAKE_HIP_COMPILER} (arch: ${CMAKE_HIP_ARCHITECTURES})")
        elseif(GPU_BACKEND STREQUAL "HIP")
            message(FATAL_ERROR "GPU_BACKEND=HIP requested but hipcc not found in PATH.")
        else()
            message(STATUS "HIP not found")
        endif()
    endif()

    # ---- CUDA ---- (only if HIP not already chosen in AUTO mode)
    if(NOT HIP_AVAILABLE AND (GPU_BACKEND STREQUAL "CUDA" OR GPU_BACKEND STREQUAL "AUTO"))
        check_language(CUDA)
        if(CMAKE_CUDA_COMPILER)
            enable_language(CUDA)
            set(CUDA_AVAILABLE TRUE)
            add_compile_definitions(CUDA_ENABLED)
            set(CMAKE_CUDA_STANDARD 17)
            find_package(CUDAToolkit REQUIRED)
            message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER} (archs: ${CMAKE_CUDA_ARCHITECTURES})")
        elseif(GPU_BACKEND STREQUAL "CUDA")
            message(FATAL_ERROR "GPU_BACKEND=CUDA requested but nvcc not found in PATH.")
        else()
            message(STATUS "CUDA not found")
        endif()
    endif()

endif()

if(NOT CUDA_AVAILABLE AND NOT HIP_AVAILABLE)
    message(STATUS "GPU_BACKEND=CPU — building CPU-only (OpenMP path)")
endif()

# ---- Dependency Resolution ----
find_package(PkgConfig REQUIRED)

list(APPEND CMAKE_PREFIX_PATH "/usr/lib64/cmake/Qt5" "/usr/lib/cmake/Qt5")
find_package(Qt5 QUIET COMPONENTS Widgets OpenGL)
if(NOT Qt5_FOUND)
    pkg_check_modules(QT5_COMPONENTS QUIET Qt5Widgets Qt5OpenGL)
    if(QT5_COMPONENTS_FOUND)
        message(STATUS "Qt5 found via pkg-config")
        set(Qt5_FOUND TRUE)
    else()
        message(FATAL_ERROR "Qt5 not found. Install qt5-qtbase-devel.")
    endif()
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

pkg_check_modules(FREENECT QUIET libfreenect)
if(FREENECT_FOUND)
    add_compile_definitions(HAVE_FREENECT)
    message(STATUS "libfreenect found — Kinect capture enabled")
else()
    message(WARNING "libfreenect not found — building without Kinect capture")
endif()

find_package(OpenGL REQUIRED)
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/eigen")
        set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/eigen")
    elseif(EXISTS "/opt/homebrew/include/eigen3/Eigen")
        set(EIGEN3_INCLUDE_DIR "/opt/homebrew/include/eigen3")
    else()
        message(FATAL_ERROR "Eigen3 not found. Run 'bash scripts/fetch_deps.sh'.")
    endif()
endif()
find_package(OpenMP)
find_package(Threads REQUIRED)

set(TINYGLTF_DIR "${CMAKE_SOURCE_DIR}/third_party/tinygltf")
if(NOT EXISTS "${TINYGLTF_DIR}/tiny_gltf.h")
    message(FATAL_ERROR "tinygltf missing. Run 'bash scripts/fetch_deps.sh'.")
endif()

# ---- Source Organization ----
file(GLOB_RECURSE SENSOR_SOURCES    CONFIGURE_DEPENDS "src/sensor/*.cpp")
file(GLOB_RECURSE TRACKING_SOURCES  CONFIGURE_DEPENDS "src/tracking/*.cpp")
file(GLOB_RECURSE TSDF_SOURCES      CONFIGURE_DEPENDS "src/tsdf/*.cpp")
file(GLOB_RECURSE MESHING_SOURCES   CONFIGURE_DEPENDS "src/meshing/*.cpp")
file(GLOB_RECURSE RENDERING_SOURCES CONFIGURE_DEPENDS "src/rendering/*.cpp")
file(GLOB_RECURSE EXPORT_SOURCES    CONFIGURE_DEPENDS "src/export/*.cpp")
file(GLOB_RECURSE GUI_SOURCES       CONFIGURE_DEPENDS "src/gui/*.cpp")
file(GLOB_RECURSE APP_SOURCES       CONFIGURE_DEPENDS "src/app/*.cpp")
file(GLOB_RECURSE UTILS_SOURCES     CONFIGURE_DEPENDS "src/utils/*.cpp")
file(GLOB_RECURSE HEADERS           CONFIGURE_DEPENDS "include/*.h")

set(ALL_SOURCES
    src/main.cpp
    ${HEADERS}
    ${SENSOR_SOURCES}
    ${TRACKING_SOURCES}
    ${TSDF_SOURCES}
    ${MESHING_SOURCES}
    ${RENDERING_SOURCES}
    ${EXPORT_SOURCES}
    ${GUI_SOURCES}
    ${APP_SOURCES}
    ${UTILS_SOURCES}
)

if(CUDA_AVAILABLE)
    file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS "src/**/*_cuda.cu")
    list(APPEND ALL_SOURCES ${CUDA_SOURCES})
endif()

if(HIP_AVAILABLE)
    file(GLOB_RECURSE HIP_SOURCES CONFIGURE_DEPENDS "src/**/*_hip.hip")
    list(APPEND ALL_SOURCES ${HIP_SOURCES})
endif()

# ---- Signal back-end label (for diagnostics) ----
set(SIGNAL_BACKEND "CPU+OpenMP")
if(HIP_AVAILABLE)
    set(SIGNAL_BACKEND "HIP+OpenMP (AMD ROCm)")
elseif(CUDA_AVAILABLE)
    set(SIGNAL_BACKEND "CUDA+OpenMP (NVIDIA)")
endif()

# ---- Build Target ----
add_executable(KinectFusionQt ${ALL_SOURCES})

target_include_directories(KinectFusionQt PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src"
    "${TINYGLTF_DIR}"
    ${EIGEN3_INCLUDE_DIR}
    ${FREENECT_INCLUDE_DIRS}
)

if(CUDA_AVAILABLE)
    target_include_directories(KinectFusionQt PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

# ---- Link Libraries ----
target_link_libraries(KinectFusionQt PRIVATE
    Qt5::Widgets
    Qt5::OpenGL
    ${OPENGL_LIBRARIES}
    ${FREENECT_LIBRARIES}
    Threads::Threads
)

if(CUDA_AVAILABLE)
    target_link_libraries(KinectFusionQt PRIVATE cuda cudart)
    target_compile_options(KinectFusionQt PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
            -diag-suppress=20012
            -diag-suppress=20015
        >
    )
endif()

if(HIP_AVAILABLE)
    # Force host linker (g++) — HIP device code is embedded in .hip.o objects.
    # ld.lld cannot resolve CXX-compiled objects (undefined symbol: main).
    set_target_properties(KinectFusionQt PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(KinectFusionQt PRIVATE amdhip64)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(KinectFusionQt PRIVATE OpenMP::OpenMP_CXX)
endif()

if(Eigen3_FOUND)
    target_link_libraries(KinectFusionQt PRIVATE Eigen3::Eigen)
endif()

if(WIN32)
    target_link_libraries(KinectFusionQt PRIVATE winmm)
endif()

# ---- Diagnostic Summary ----
message(STATUS "============================================")
message(STATUS "KinectFusionQt Configuration:")
message(STATUS "  GPU_BACKEND:  ${GPU_BACKEND}")
message(STATUS "  Signal Path:  ${SIGNAL_BACKEND}")
message(STATUS "  Build Type:   ${CMAKE_BUILD_TYPE}")
if(CUDA_AVAILABLE)
    message(STATUS "  CUDA Arch:    ${CMAKE_CUDA_ARCHITECTURES}")
endif()
if(HIP_AVAILABLE)
    message(STATUS "  HIP Arch:     ${CMAKE_HIP_ARCHITECTURES}")
endif()
message(STATUS "  OpenMP:       ${OpenMP_CXX_FOUND}")
message(STATUS "  Kinect I/O:   ${KINECT_IO_SUMMARY}")
message(STATUS "  Qt Version:   ${Qt6_VERSION}")
message(STATUS "============================================")
