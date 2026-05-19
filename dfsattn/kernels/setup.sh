#!/bin/bash
# Setup script for building fast CUDA kernels

set -e  # Exit on error

echo "Checking dependencies..."

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo "ERROR: CMake is not installed!"
    echo ""
    echo "Please install CMake:"
    echo "  Ubuntu/Debian: sudo apt install cmake"
    echo "  Or download from: https://cmake.org/download/"
    exit 1
fi

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "WARNING: nvcc (CUDA compiler) not found in PATH"
    echo "Make sure CUDA is installed and in your PATH"
fi

# Check for Python and PyTorch
if ! python -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch is not installed!"
    exit 1
fi

# Check for required kernel source files
if [ ! -d "csrc" ] || [ ! -d "include" ]; then
    echo "ERROR: Kernel source files not found!"
    exit 1
fi

if [ ! -d "3rdparty" ]; then
    echo "ERROR: Third-party kernel headers not found!"
    echo "Expected cutlass, flashinfer, and pybind under dfsattn/kernels/3rdparty/."
    exit 1
fi

echo "Dependencies check passed!"
echo ""

# if nvjitlink not in LD_LIBRARY_PATH, add it
if [[ ":$LD_LIBRARY_PATH:" != *":$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/nvjitlink/lib')"):"* ]]; then
    export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/nvjitlink/lib')"):$LD_LIBRARY_PATH
fi

echo "Building kernels..."
mkdir -p build
cd build

echo "Running CMake..."
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` -DUSE_SYSTEM_NVTX:BOOL=ON ..

if [ $? -ne 0 ]; then
    echo "ERROR: CMake configuration failed!"
    exit 1
fi

echo "Compiling..."
make -j

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi

# Check for the built module (it may have a version suffix like .cpython-310-x86_64-linux-gnu.so)
if ls _kernels*.so 1> /dev/null 2>&1; then
    KERNEL_FILE=$(ls _kernels*.so | head -1)
    echo ""
    echo "✓ Build complete! The _kernels module is: build/$KERNEL_FILE"
    echo ""
    echo "You can verify the installation with:"
    echo "  python -c 'from dfsattn.kernels import ENABLE_FAST_KERNEL; print(f\"Fast kernels: {ENABLE_FAST_KERNEL}\")'"
else
    echo "WARNING: _kernels*.so not found after build. Check build errors above."
    exit 1
fi
