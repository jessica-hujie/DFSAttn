# DFSAttn: Dynamic Fine-grained Sparse Attention for Efficient Video Generation

DFSAttn is a training-free sparse attention framework for efficient video
diffusion inference. It exploits dynamic and fine-grained sparsity in diffusion
transformers through 3D Hilbert token reordering, hierarchical block scoring,
and adaptive sparse mask caching, achieving up to 2.1x end-to-end speedup while
preserving generation quality.

[Paper](https://arxiv.org/abs/2605.23445) | [Examples](#examples) | [Installation](#installation) | [Citation](#citation)

## Highlights

- Training-free acceleration for video diffusion transformers.
- Fine-grained sparsity with GPU-friendly block-sparse execution.
- 3D Hilbert token reordering to preserve spatiotemporal locality.
- Hierarchical block scoring for more accurate sparse mask selection.
- Adaptive sparse mask caching across denoising steps.
- Supports HunyuanVideo-T2V-13B and Wan2.1-T2V-14B.

## What Is Included

```text
.
├── dfsattn/                    # DFSAttn attention processors and utilities
│   ├── attention_hyvideo.py     # DFS attention for HunyuanVideo
│   ├── attention_wan.py         # DFS attention for Wan
│   ├── replace_hyvideo.py       # HunyuanVideo attention replacement
│   ├── replace_wan.py           # Wan attention replacement
│   ├── fullattention.py         # Dense attention fallback
│   ├── utils/                   # Seed, logging, ordering, visualization helpers
│   └── kernels/                 # Optional CUDA/Triton acceleration kernels
├── hyvideo_t2v_inference.py     # HunyuanVideo text-to-video inference
├── wan21_t2v_inference.py       # Wan 2.1 text-to-video inference
├── *_720p_dfs.sh                # Batch DFSAttn launch examples
├── dataloader.py                # Prompt loading helpers
├── requirements.txt             # Reference Python dependencies
├── examples/prompts.txt         # Minimal prompt file for smoke tests
└── assets/examples/             # Example full-attention and DFSAttn videos
```

## Installation

The code targets Linux CUDA environments. The reference environment uses Python
3.10, PyTorch 2.6, CUDA 12.x, Diffusers, FlashAttention, and
`block_sparse_attn`.

### 1. Create Environment

Install the PyTorch/CUDA pair that matches your machine first, then install the
CUDA extension build tools:

```bash
conda create -n dfsattn python=3.10 -y
conda activate dfsattn
# Example only. Pick the PyTorch command that matches your CUDA runtime.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -U setuptools wheel cmake ninja psutil packaging
```

### 2. Required: Block-Sparse-Attention

DFSAttn directly uses the block-sparse attention operator from
[MIT HAN Lab Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention).
Build it from the upstream repository after PyTorch is installed. `CUDA_HOME`
must point to the same CUDA toolkit version used by PyTorch; for example,
PyTorch `cu124` should use CUDA 12.4 `nvcc`.

```bash
git clone https://github.com/mit-han-lab/Block-Sparse-Attention.git
cd Block-Sparse-Attention
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
# A100: 80, A10/RTX 30xx: 86, H100/H200: 90. See the upstream repo for more architectures.
export BLOCK_SPARSE_ATTN_CUDA_ARCHS=80
pip install --no-build-isolation .
cd ..
```

### 3. Install DFSAttn Dependencies

```bash
pip install -r requirements.txt
```

The versions in `requirements.txt` are pinned where Wan 2.1 depends on a
specific Diffusers/Transformers API. Upgrade those packages only after checking
the attention processor interfaces.

### 4. Optional: Fast QK-Norm and RoPE Kernels

DFSAttn can integrate the fast QK-Norm and RoPE CUDA/Triton kernels from
[Sparse-VideoGen](https://github.com/svg-project/Sparse-VideoGen). These
kernels are optional, but recommended when reproducing the end-to-end speedups
reported in the paper. DFSAttn falls back to PyTorch implementations if they are
not built.

Prepare the kernel sources and third-party headers by following the upstream
Sparse-VideoGen customized-kernel setup first. This initializes the required
submodules and verifies that the upstream kernels build in your environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/svg-project/Sparse-VideoGen.git
cd Sparse-VideoGen
pip install -U setuptools
git submodule update --init --recursive
cd svg/kernels
pip install -U cmake
bash setup.sh
```

DFSAttn expects the initialized Sparse-VideoGen third-party layout at:

```text
dfsattn/kernels/3rdparty/
├── cutlass/
├── flashinfer/
└── pybind/
```

```bash
cd /path/to/DFSAttn
rm -rf dfsattn/kernels/3rdparty
cp -a /path/to/Sparse-VideoGen/svg/kernels/3rdparty dfsattn/kernels/3rdparty
```

Then build the DFSAttn extension:

```bash
cd dfsattn/kernels
bash setup.sh
python -c 'from dfsattn.kernels import ENABLE_FAST_KERNEL; print(ENABLE_FAST_KERNEL)'
```

## Model Checkpoints

Use either Hugging Face model IDs or local checkpoint directories:

```bash
export HYVIDEO_MODEL_ID=/path/to/HunyuanVideo
export WAN_MODEL_ID=/path/to/Wan2.1-T2V-14B
```

You can also pass `--model_id` directly to the inference scripts.

## Single Prompt Inference

Run HunyuanVideo with DFSAttn:

```bash
python hyvideo_t2v_inference.py \
  --model_id "$HYVIDEO_MODEL_ID" \
  --prompt "A cinematic city street at night with colorful neon lights" \
  --output_file output/hyvideo_dfs.mp4 \
  --mode dfs \
  --sparsity 0.3 \
  --tile_size 16 \
  --block_size 128 \
  --order hilbert3d \
  --skip_steps 12 \
  --cache_interval 12 \
  --sparsity_dcrt 0.1
```

Run Wan 2.1 with DFSAttn:

```bash
python wan21_t2v_inference.py \
  --model_id "$WAN_MODEL_ID" \
  --prompt "A cat walks on the grass, realistic" \
  --output_file output/wan_dfs.mp4 \
  --mode dfs \
  --sparsity 0.3 \
  --tile_size 16 \
  --block_size 128 \
  --order hilbert3d \
  --skip_steps 12 \
  --cache_interval 12 \
  --sparsity_dcrt 0.1
```

## Batch Inference

The launch scripts are configured through environment variables and do not depend on local absolute paths.

```bash
CUDA_VISIBLE_DEVICES=0 \
HYVIDEO_MODEL_ID=/path/to/HunyuanVideo \
PROMPT_FILE=examples/prompts.txt \
OUTPUT_DIR=output/hyvideo/dfs \
./hyvideo_t2v_720p_dfs.sh
```

```bash
CUDA_VISIBLE_DEVICES=0 \
WAN_MODEL_ID=/path/to/Wan2.1-T2V-14B \
PROMPT_FILE=examples/prompts.txt \
OUTPUT_DIR=output/wan/dfs \
./wan21_t2v_720p_dfs.sh
```

Useful script variables:

- `PROMPT_FILE`: text file with one prompt per line.
- `OUTPUT_DIR`: directory for generated videos.
- `START_IDX` and `END_IDX`: inclusive prompt index range.
- `SPARSITY`, `SKIP_STEPS`, `CACHE_INTERVAL`, `SPARSITY_DCRT`, `TILE_SIZE`, `BLOCK_SIZE`, `ORDER`: DFSAttn parameters.
- `HEIGHT`, `WIDTH`, `NUM_FRAMES`, `NUM_INFERENCE_STEPS`, `SEED`: generation settings.

## Default Parameters

| Argument | Default | Description |
|---|---:|---|
| `SPARSITY` / `--sparsity` | `0.3` | Initial sparsity budget after skipped denoising steps. |
| `SKIP_STEPS` / `--skip_steps` | `12` | Number of early denoising steps using full attention. |
| `CACHE_INTERVAL` / `--cache_interval` | `12` | Sparse mask refresh interval. |
| `SPARSITY_DCRT` / `--sparsity_dcrt` | `0.1` | Sparsity decrease after each refresh. |
| `TILE_SIZE` / `--tile_size` | `16` | Token grouping granularity for hierarchical scoring. |
| `BLOCK_SIZE` / `--block_size` | `128` | Block size used by block-sparse attention. |
| `ORDER` / `--order` | `hilbert3d` | Token ordering strategy. |

DFSAttn runs full attention for the first `SKIP_STEPS` diffusion steps. After
that, it starts from `SPARSITY` and refreshes the sparse mask every
`CACHE_INTERVAL` diffusion steps, decreasing sparsity by `SPARSITY_DCRT` each
interval. If a scheduled refresh would make sparsity non-positive, DFSAttn skips
that refresh and keeps using the previous cached mask and sparsity. The default
`SKIP_STEPS=12`, `CACHE_INTERVAL=12`, `SPARSITY_DCRT=0.1` gives HyVideo refresh
steps `12`, `24`, and `36`; the next scheduled step `48` would be `0.0`, so it
reuses the `36` mask. Wan internally visits each diffusion step twice, so the
same defaults map to internal refresh steps `24`, `48`, and `72`; the scheduled
`96` refresh is skipped.

## Examples

| Example | Full Attention | DFSAttn |
|---|---|---|
| 1 | [dense_1.mp4](assets/examples/dense_1.mp4) | [dfs_1.mp4](assets/examples/dfs_1.mp4) |
| 2 | [dense_2.mp4](assets/examples/dense_2.mp4) | [dfs_2.mp4](assets/examples/dfs_2.mp4) |

## Troubleshooting

- Make sure `CUDA_HOME` points to the CUDA toolkit matching your PyTorch CUDA version.
- If `block_sparse_attn` fails to compile, set `BLOCK_SPARSE_ATTN_CUDA_ARCHS` according to your GPU architecture.
- If optional fast kernels are unavailable, DFSAttn falls back to PyTorch implementations.
- If Wan 2.1 fails after upgrading Diffusers or Transformers, use the versions pinned in `requirements.txt`.

## Citation

If you find DFSAttn useful for your research, please cite:

```bibtex
@article{hu2026dfsattn,
  title={DFSAttn: Dynamic Fine-grained Sparse Attention for Efficient Video Generation},
  author={Hu, Jie and Gao, Zixiang and He, Yutong and Yuan, Kun},
  journal={arXiv preprint arXiv:2605.23445},
  year={2026}
}
```

## Acknowledgements

- DFSAttn uses the block-sparse attention operator from [MIT HAN Lab Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention), which exposes the `block_sparse_attn_func` interface. Please preserve the upstream BSD-3-Clause license notice when redistributing adapted code.
- For end-to-end acceleration, DFSAttn can integrate fast QK-Norm and RoPE kernels from [Sparse-VideoGen](https://github.com/svg-project/Sparse-VideoGen), which provides customized kernels for faster video diffusion inference.

## License

Please follow the license terms of this repository and the upstream projects
listed above.
