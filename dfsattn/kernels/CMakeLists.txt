cmake_minimum_required(VERSION 3.26.4)

# Adjust compiler paths if needed
# set(CMAKE_C_COMPILER "/usr/bin/gcc-11")
# set(CMAKE_CXX_COMPILER "/usr/bin/g++-11")
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Adjust CUDA architecture for your GPU:
# - 80 for A100
# - 89 for H100  
# - 90a for H200
# - 75 for T4, 86 for A10, etc.
set(CMAKE_CUDA_ARCHITECTURES 80)

project(_kernels LANGUAGES CUDA CXX)

# Check: https://stackoverflow.com/questions/68401650/how-can-i-make-a-pytorch-extension-with-cmake
# Fix linking error: https://github.com/pytorch/pytorch/issues/108041
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

# Try combine pybind
# Check: https://qiita.com/syoyo/items/c3e8e6e5c3e2d69c2325
add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty/pybind ${CMAKE_BINARY_DIR}/pybind11)

file(GLOB PYTORCH_SOURCES "csrc/*.cu")
pybind11_add_module(_kernels MODULE ${PYTORCH_SOURCES})

# Add dependency libraries
target_include_directories(_kernels PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/cutlass/include)
target_include_directories(_kernels PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/pybind/include)
# flashinfer is required for rope_enc.cuh
target_include_directories(_kernels PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/flashinfer/include)

# Add customized kernel impl
target_include_directories(_kernels PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_compile_options(_kernels PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-extended-lambda 
        --expt-relaxed-constexpr
        --use_fast_math
        --disable-warnings
    >
    $<$<COMPILE_LANGUAGE:CXX>:-w>
)
target_link_libraries(_kernels PRIVATE ${TORCH_LIBRARIES} Python::Python pybind11::module ${TORCH_PYTHON_LIBRARY})
