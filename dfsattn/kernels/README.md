# Fast CUDA/Triton Kernels for DFS Attention

This directory integrates optimized CUDA/Triton kernels for accelerating DFSAttn.
The fast QK-Norm and RoPE kernels come from the customized kernel implementation
in [Sparse-VideoGen](https://github.com/svg-project/Sparse-VideoGen).

## Quick Start

1. **Prepare kernel sources and third-party headers** using the upstream
   [Sparse-VideoGen](https://github.com/svg-project/Sparse-VideoGen)
   `svg/kernels/` layout. DFSAttn expects:

   ```bash
   git clone https://github.com/svg-project/Sparse-VideoGen.git
   ```

   ```text
   dfsattn/kernels/3rdparty/
   ├── cutlass/
   ├── flashinfer/
   └── pybind/
   ```

2. **Adjust CUDA architecture** in `CMakeLists.txt` to match your GPU:
   - A100: `80`
   - H100: `89`
   - H200: `90a`
   - T4: `75`
   - A10: `86`

3. **Build the kernels**:
   ```bash
   cd fastvideo/dfsattn/kernels
   chmod +x setup.sh
   ./setup.sh
   ```

4. **Verify installation**:
   ```python
   from dfsattn.kernels import ENABLE_FAST_KERNEL
   print(f"Fast kernels enabled: {ENABLE_FAST_KERNEL}")
   ```

## What's Included

- **RMSNorm**: Fast RMS normalization for QK normalization (~1-2s speedup)
- **RoPE**: Fast Rotary Position Embedding application (~0.5-1s speedup)

## Performance

Expected speedup: **~2-3 seconds per inference step** when using full attention.

## Files

- `__init__.py`: Python wrapper that imports kernels or falls back to PyTorch
- `CMakeLists.txt`: CMake build configuration
- `setup.sh`: Build script
- `INTEGRATION_GUIDE.md`: Detailed integration instructions

## Troubleshooting

See `INTEGRATION_GUIDE.md` for detailed troubleshooting steps.
