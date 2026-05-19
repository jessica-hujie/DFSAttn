import argparse
import os
import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel, FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video
import time
from dfsattn.utils.seed import seed_everything
from dfsattn.utils.logger import logger
from dfsattn.utils.order import block3d_perm, hwf, fwh, hilbert3d_perm, hilbert2d_perm
from dataloader import load_prompt_or_image
from dfsattn.replace_hyvideo import HunyuanVideo_DFSAttn_Processor2_0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=os.getenv("HYVIDEO_MODEL_ID"), help="Model ID or local checkpoint path to use for generation")
    parser.add_argument("--height", type=int, default=720, help="Height of the generated video")
    parser.add_argument("--width", type=int, default=1280, help="Width of the generated video")
    parser.add_argument("--num_frames", type=int, default=129, help="Number of frames in the generated video")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps in the generated video")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--negative_prompt", type=str, default="Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion", help="Negative text prompt to avoid certain features")
    parser.add_argument("--prompt", type=str, default="As night falls, on Michigan Avenue in Chicago, the towering buildings are decorated with colorful neon lights on their exterior walls, forming a brilliant ocean of light that contrasts sharply with the dark night sky.", help="Text prompt for video generation")

    parser.add_argument("--prompt_source", type=str, default="prompt", choices=["prompt", "T2V_Hyv_VBench", "T2V_Hyv_Web", "T2V_Xingyang_Motion", "T2V_Xingyang_VBench"], help="Source of the prompt")
    parser.add_argument("--prompt_idx", type=int, default=0, help="Index of the prompt")
    parser.add_argument("--output_file", type=str, default="output.mp4", help="Output video file name")

    parser.add_argument("--mode", type=str, default="dfs", choices=["dfs", "flash", "torch", "vanilla"])
    parser.add_argument("--sparsity", type=float, default=0.3, help="The sparsity of sparse attention pattern.")
    parser.add_argument("--tile_size", type=int, default=16, help="The tile size of pooling in dfs attention.")
    parser.add_argument("--block_size", type=int, default=128, help="The block size in dfs attention.")
    parser.add_argument("--order", type=str, default="hilbert3d", choices=["org", "hilbert2d", "blk", "hwf", "fwh","hilbert3d"])
    parser.add_argument("--skip_layers", type=list[int], default=[], help="Layer indices to skip in dfs attention")
    parser.add_argument("--skip_steps", type=int, default=12, help="Number of steps to skip in dfs attention")
    parser.add_argument("--cache_interval", type=int, default=12, help="Diffusion-step interval between sparse mask refreshes")
    parser.add_argument("--sparsity_dcrt", type=float, default=0.1, help="Sparsity decrement applied every cache interval")
    parser.add_argument("--cache_flag", type=bool, default=True, help="Cache the sparse mask in dfs attention")
   
    args = parser.parse_args()
    if args.model_id is None:
        parser.error("Please pass --model_id or set HYVIDEO_MODEL_ID.")
    return args


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)

    transformer = HunyuanVideoTransformer3DModel.from_pretrained(args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
    flow_shift = 7.0
    scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
    pipe = HunyuanVideoPipeline.from_pretrained(args.model_id, transformer=transformer, scheduler=scheduler, torch_dtype=torch.bfloat16)
    pipe.vae.enable_tiling()
    pipe.to("cuda")

    args.prompt, _ = load_prompt_or_image(args.prompt_source, args.prompt_idx, args.prompt, None)

    latent_f, latent_h, latent_w= args.num_frames // 4 + 1, args.height // 16, args.width // 16
    video_len = latent_f * latent_h * latent_w
    if args.order == "org":
        video_perm = None
    elif args.order == "hilbert2d":
        video_perm = hilbert2d_perm(latent_f, latent_h, latent_w)
    elif args.order == "blk":
        video_perm = block3d_perm(latent_f, latent_h, latent_w, a=4, b=4, c=4)
    elif args.order == "hwf":
        video_perm = hwf(latent_f, latent_h, latent_w)
    elif args.order == "fwh":
        video_perm = fwh(latent_f, latent_h, latent_w)
    elif args.order == "hilbert3d":
        video_perm = hilbert3d_perm(latent_f, latent_h, latent_w)

    attn_processors = {}
    processors_id = 0
    for k,v in transformer.attn_processors.items():
        if "token_refiner" in k:
            attn_processors[k] = v
        else:
            attn_processors[k] = HunyuanVideo_DFSAttn_Processor2_0(
                args.mode,
                args.sparsity,
                args.tile_size,
                args.block_size,
                video_len,
                video_perm,
                processors_id,
                args.skip_layers,
                args.skip_steps,
                args.cache_interval,
                args.sparsity_dcrt,
                args.cache_flag,
            )
            processors_id += 1   
    transformer.set_attn_processor(attn_processors)

    total_start_time = time.time()

    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=6.0,
        num_inference_steps=args.num_inference_steps,
    ).frames[0]

    total_end_time = time.time()
    total_generation_time = total_end_time - total_start_time
    
    logger.info(f"Total generation time: {total_generation_time:.2f} s")

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    export_to_video(output, args.output_file, fps=24)

    
