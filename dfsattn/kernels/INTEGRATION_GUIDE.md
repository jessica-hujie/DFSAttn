# Fast CUDA/Triton Kernels Integration Guide

This guide explains how to build the optional fast CUDA/Triton kernels for
DFSAttn. These fast QK-Norm and RoPE kernels are integrated from the customized
kernel implementation in
[Sparse-VideoGen](https://github.com/svg-project/Sparse-VideoGen).

## Overview

The fast kernels provide optimized implementations of:
1. **RMSNorm**: Fast RMS normalization for QK normalization
2. **RoPE**: Fast Rotary Position Embedding application

These kernels can provide ~2-3s speedup per inference step.

## Step 1: Prepare Kernel Sources From Sparse-VideoGen

Use [Sparse-VideoGen](https://github.com/svg-project/Sparse-VideoGen) as the
upstream source for the fast kernel implementation. DFSAttn expects the same
kernel source and third-party dependency layout as Sparse-VideoGen's
`svg/kernels/` directory:

```bash
git clone https://github.com/svg-project/Sparse-VideoGen.git
```

```
fastvideo/dfsattn/kernels/
├── csrc/
│   ├── ops.cu
│   ├── ops.h
│   └── pytorch_extension_utils.h
├── include/
│   ├── norm/
│   │   ├── narrow_layer_norm.cuh
│   │   ├── narrow_rms_norm.cuh
│   │   ├── cutlass_layer_norm.cuh
│   │   ├── cutlass_rms_norm.h
│   │   └── device_utils.cuh
│   └── rope/
│       ├── rope_enc.cuh
│       ├── rope_enc_txtlast.cuh
│       └── rope_enc_complex.cuh
└── 3rdparty/
    ├── cutlass/
    ├── flashinfer/
    └── pybind/
```

## Step 2: Create CMakeLists.txt

The `CMakeLists.txt` file is already created in `fastvideo/dfsattn/kernels/CMakeLists.txt`.
You may need to adjust:
- CUDA architecture: change `CMAKE_CUDA_ARCHITECTURES` to match your GPU
  (for example, `80` for A100 or `89` for H100)
- Compiler paths: adjust the optional compiler settings if your environment
  requires a specific GCC/G++ version

## Step 3: Build the Kernels

Run the setup script:

```bash
cd fastvideo/dfsattn/kernels
bash setup.sh
```

Or manually:
```bash
mkdir -p build
cd build
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` -DUSE_SYSTEM_NVTX:BOOL=ON ..
make -j
```

## Step 4: Verify Installation

Test that the kernels can be imported:

```python
from dfsattn.kernels import ENABLE_FAST_KERNEL, apply_qk_norm
print(f"Fast kernels enabled: {ENABLE_FAST_KERNEL}")
```

## Step 5: Usage in Code

The kernels are automatically used when available. The `replace_hyvideo.py` file has been modified to:
1. Try importing fast kernels
2. Fall back to PyTorch implementation if kernels are not available
3. Use fast kernels for QK normalization and RoPE when enabled

## Troubleshooting

### Build Errors

1. **CUDA not found**: Ensure CUDA is in PATH and LD_LIBRARY_PATH
2. **PyTorch not found**: Ensure PyTorch is installed with CUDA support
3. **CUTLASS not found**: Prepare the `3rdparty/cutlass` directory from the
   Sparse-VideoGen kernel layout

### Runtime Errors

1. **ImportError**: Check that `build/_kernels.so` exists and is in the Python path
2. **CUDA errors**: Ensure GPU is available and CUDA version matches PyTorch
3. **Shape mismatches**: Ensure tensors are contiguous before calling kernels

### Performance

If you don't see speedup:
1. Verify `ENABLE_FAST_KERNEL = True` in logs
2. Check that tensors are contiguous (kernels require this)
3. Profile to ensure kernels are actually being called

## Dependencies

- CUDA toolkit (11.8+)
- CMake (3.26+)
- PyTorch with CUDA support
- Sparse-VideoGen kernel sources:
  [svg/kernels](https://github.com/svg-project/Sparse-VideoGen/tree/main/svg/kernels)
- CUTLASS under `3rdparty/cutlass`
- FlashInfer headers under `3rdparty/flashinfer`
- pybind11 under `3rdparty/pybind`
