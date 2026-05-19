#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"

sparsity="${SPARSITY:-0.3}"
skip_steps="${SKIP_STEPS:-12}"
cache_interval="${CACHE_INTERVAL:-12}"
sparsity_dcrt="${SPARSITY_DCRT:-0.1}"
tile_size="${TILE_SIZE:-16}"
block_size="${BLOCK_SIZE:-128}"
order="${ORDER:-hilbert3d}"
prompt_file="${PROMPT_FILE:-${REPO_ROOT}/examples/prompts.txt}"
output_dir="${OUTPUT_DIR:-${REPO_ROOT}/output/wan/dfs}"
start_idx="${START_IDX:-0}"
seed="${SEED:-42}"
height="${HEIGHT:-720}"
width="${WIDTH:-1280}"
num_frames="${NUM_FRAMES:-81}"
num_inference_steps="${NUM_INFERENCE_STEPS:-50}"
model_id="${WAN_MODEL_ID:-}"

if [ -z "$model_id" ]; then
    echo "ERROR: Set WAN_MODEL_ID."
    exit 1
fi

if [ ! -f "$prompt_file" ]; then
    echo "ERROR: Prompt file not found: $prompt_file"
    exit 1
fi

num_prompts=$(wc -l < "$prompt_file")
end_idx="${END_IDX:-$((num_prompts - 1))}"

echo "Found $num_prompts prompts in $prompt_file"
echo "Generating videos for prompts ${start_idx} to ${end_idx}"
mkdir -p "$output_dir"

for prompt_idx in $(seq "$start_idx" "$end_idx"); do
    echo "Processing prompt $prompt_idx..."

    python "${REPO_ROOT}/wan21_t2v_inference.py" \
        --model_id "$model_id" \
        --seed "$seed" \
        --height "$height" \
        --width "$width" \
        --num_frames "$num_frames" \
        --num_inference_steps "$num_inference_steps" \
        --prompt "$prompt_file" \
        --prompt_source "T2V_Wan_VBench" \
        --prompt_idx "$prompt_idx" \
        --output_file "${output_dir}/${prompt_idx}.mp4" \
        --mode "dfs" \
        --sparsity "$sparsity" \
        --tile_size "$tile_size" \
        --block_size "$block_size" \
        --skip_steps "$skip_steps" \
        --cache_interval "$cache_interval" \
        --sparsity_dcrt "$sparsity_dcrt" \
        --order "$order" \
        --cache_flag True

    echo "Successfully generated video for prompt $prompt_idx"
done

echo "Finished processing all prompts"
