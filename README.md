# DFSAttn

DFSAttn is a sparse attention method for text-to-video diffusion inference. This repository contains the DFSAttn implementation and runnable inference entry points for HunyuanVideo and Wan 2.1.

The public release is centered on `dfsattn/`. Other local research baselines such as SVG and radial attention are intentionally not part of this repository.

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
├── eval/vbench/                 # Optional VBench prompt utilities
├── requirements.txt             # Reference Python dependencies
└── examples/prompts.txt         # Minimal prompt file for smoke tests
```

## Environment

The code targets Linux CUDA environments. The reference environment uses Python
3.10, PyTorch 2.6, CUDA 12.x, Diffusers, FlashAttention, and
`block_sparse_attn`. Install the PyTorch/CUDA pair that matches your machine
first, then install the Python dependencies:

```bash
conda create -n dfsattn python=3.10 -y
conda activate dfsattn
# Example only. Pick the PyTorch command that matches your CUDA runtime.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

DFSAttn directly uses the block-sparse attention operator from
[MIT HAN Lab Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention).
Install it from PyPI or follow the upstream repository for source builds and
CUDA compatibility notes:

```bash
pip install block_sparse_attn
```

```bash
git clone https://github.com/mit-han-lab/Block-Sparse-Attention.git
cd Block-Sparse-Attention
pip install .
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

## Optional Fast Kernels

`dfsattn/kernels/` integrates the fast QK-Norm and RoPE CUDA/Triton kernels
from [Sparse-VideoGen](https://github.com/svg-project/Sparse-VideoGen). DFSAttn
falls back to PyTorch implementations if these kernels are not built.

Prepare the kernel sources and third-party headers following the upstream
Sparse-VideoGen kernel layout (`svg/kernels/`). The expected local layout is:

```text
dfsattn/kernels/3rdparty/
├── cutlass/
├── flashinfer/
└── pybind/
```

For a source-based setup, clone Sparse-VideoGen and use its `svg/kernels`
directory as the upstream reference:

```bash
git clone https://github.com/svg-project/Sparse-VideoGen.git
```

Then build the DFSAttn extension:

```bash
cd dfsattn/kernels
./setup.sh
```

## Notes For Release

- Generated videos, logs, caches, local checkpoints, and build products are ignored by `.gitignore`.
- The vendored `Block-Sparse-Attention/` checkout is not needed in the public repository if `block_sparse_attn` is installed as a dependency.
- Add a top-level `LICENSE` before publishing if you want others to have explicit reuse rights.

## Acknowledgements

- The optional fast QK-Norm and RoPE kernels are adapted from [Sparse-VideoGen](https://github.com/svg-project/Sparse-VideoGen), which provides customized kernels for faster video diffusion inference.
- The block-sparse attention operator used by DFSAttn is adapted from [MIT HAN Lab Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention), which exposes the `block_sparse_attn_func` interface. Please preserve the upstream BSD-3-Clause license notice when redistributing adapted code.
