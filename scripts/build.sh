#!/usr/bin/env bash
# ============================================================
# scripts/build.sh — KinectFusionQt build helper
#
# Usage:
#   ./scripts/build.sh [--hip|--cuda|--cpu] [--debug] [--clean] [-jN] [--arch=A.B]
#
# Options:
#   --hip     Force HIP/ROCm back-end (AMD GPU)
#   --cuda    Force CUDA back-end (NVIDIA GPU)
#   --cpu     Force CPU-only (OpenMP) back-end
#   (none)    AUTO: prefer HIP, fall back to CUDA, then CPU
#   --debug   Debug build (no optimisations)
#   --clean   Wipe the build directory before configuring
#   -jN       Parallelism, no space (default: min(nproc, 28))
# ============================================================
set -euo pipefail

# KIN-FORK: this box installs libfreenect.pc under /usr/local, absent from
# Gentoo's default pkg-config search path.
export PKG_CONFIG_PATH="/usr/local/lib64/pkgconfig:/usr/local/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

BACKEND="AUTO"
BUILD_TYPE="Release"
CLEAN=0
JOBS=$(( $(nproc) > 28 ? 28 : $(nproc) ))
EXTRA_CMAKE_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --hip)    BACKEND="HIP"   ;;
        --cuda)   BACKEND="CUDA"  ;;
        --cpu)    BACKEND="CPU"   ;;
        --debug)  BUILD_TYPE="Debug" ;;
        --clean)  CLEAN=1         ;;
        -j*)      JOBS="${arg#-j}" ;;
        --arch=*) EXTRA_CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="${arg#--arch=}") ;;
        *)  echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# Map backend to build directory
case "$BACKEND" in
    HIP)  BUILD_DIR="$PROJECT_ROOT/build-hip"  ;;
    CUDA) BUILD_DIR="$PROJECT_ROOT/build-cuda" ;;
    CPU)  BUILD_DIR="$PROJECT_ROOT/build-cpu"  ;;
    AUTO) BUILD_DIR="$PROJECT_ROOT/build"      ;;
esac

echo "================================================================"
echo "  KinectFusionQt build"
echo "  Backend   : $BACKEND"
echo "  Build type: $BUILD_TYPE"
echo "  Build dir : $BUILD_DIR"
echo "  Jobs      : $JOBS"
echo "================================================================"

if [[ $CLEAN -eq 1 ]]; then
    echo "[build.sh] Cleaning $BUILD_DIR ..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$PROJECT_ROOT" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DGPU_BACKEND="$BACKEND" \
    "${EXTRA_CMAKE_ARGS[@]}"

ninja -j "$JOBS"

echo ""
echo "================================================================"
echo "  Build complete: $BUILD_DIR/KinectFusionQt"
echo "================================================================"
