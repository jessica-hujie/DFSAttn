import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention
from typing import Optional
from torch.nn.attention import sdpa_kernel, SDPBackend

from .fullattention import full_attention
from .attention_wan import dfs_attention
from dfsattn.utils.logger import logger

# Try to import fast kernels (Triton RMSNorm and CUDA RoPE)
try:
    import sys
    import os
    
    # Add build directory to path for CUDA kernels
    # __file__ is dfsattn/replace_wan.py, so kernel_dir should be dfsattn/
    kernel_dir = os.path.dirname(os.path.abspath(__file__))  # This is dfsattn/
    build_dir = os.path.join(kernel_dir, "kernels", "build")  # dfsattn/kernels/build
    if build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    
    # Try to import CUDA kernels for RoPE
    try:
        import _kernels
        # Check if the function exists
        if hasattr(_kernels, 'apply_qk_rope_inplace_cossin_complex'):
            ENABLE_FAST_ROPE = True
            
            def apply_rotary_emb_fast(query: torch.Tensor, key: torch.Tensor, freqs):
                """Fast RoPE using CUDA kernels for complex rotary embeddings."""
                # freqs should already be preprocessed to tuple format (freqs_real, freqs_imag)
                # The CUDA kernel expects real and imaginary frequency tensors.
                if isinstance(freqs, (tuple, list)) and len(freqs) == 2:
                    # Preprocessed format: (freqs_real, freqs_imag)
                    freqs_real, freqs_imag = freqs
                else:
                    # Should not happen if preprocessing is done correctly
                    raise ValueError(f"Expected freqs to be tuple (freqs_real, freqs_imag), got {type(freqs)}")
                
                _kernels.apply_qk_rope_inplace_cossin_complex(query, key, freqs_real, freqs_imag, 0)  # len_text_prompt = 0
                return query, key
        else:
            ENABLE_FAST_ROPE = False
            apply_rotary_emb_fast = None
            logger.warning("_kernels imported but apply_qk_rope_inplace_cossin_complex not found")
    except ImportError as e:
        ENABLE_FAST_ROPE = False
        apply_rotary_emb_fast = None
        logger.warning(f"Could not import _kernels for RoPE: {e}")
    except Exception as e:
        ENABLE_FAST_ROPE = False
        apply_rotary_emb_fast = None
        logger.warning(f"Error setting up fast RoPE kernel: {e}")
    
    # Try to import Triton RMSNorm
    try:
        # Try local implementation first
        from .kernels.triton.rmsnorm import triton_rmsnorm_forward
        ENABLE_FAST_NORM = True
    except ImportError:
        ENABLE_FAST_NORM = False
        triton_rmsnorm_forward = None
    
    if ENABLE_FAST_ROPE or ENABLE_FAST_NORM:
        logger.info("Fast CUDA/Triton kernels enabled for WAN (RoPE: {}, Norm: {})".format(
            ENABLE_FAST_ROPE, ENABLE_FAST_NORM))
    else:
        logger.info("Fast kernels not available for WAN, using PyTorch fallback")
        
except Exception as e:
    ENABLE_FAST_ROPE = False
    ENABLE_FAST_NORM = False
    apply_rotary_emb_fast = None
    triton_rmsnorm_forward = None
    logger.info(f"Fast kernels not available for WAN: {e}, using PyTorch fallback")

class Wan_DFSAttn_Processor2_0:

    def __init__(
        self,
        mode,
        sparsity,
        tile_size,
        block_size,
        video_perm,
        processor_id,
        skip_layers,
        skip_steps,
        cache_interval,
        sparsity_dcrt,
        cache_flag,
        cross_flag,
    ):
        self.mode = mode
        self.sparsity = sparsity
        self.tile_size = tile_size
        self.block_size = block_size
        self.video_perm = video_perm
        self.step_idx = 0
        self.layer_idx = processor_id
        self.skip_layers = skip_layers
        self.skip_steps = skip_steps
        self.cache_interval = cache_interval
        self.sparsity_dcrt = sparsity_dcrt
        self.cache_flag = cache_flag
        self.cross_flag = cross_flag

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # QK normalization - use Triton RMSNorm if available.
        # Note: Normalize BEFORE transpose, so tensors are (B, L, H*D) shape
        # triton_rmsnorm_forward handles the reshaping internally via flatten_if_batched
        # Skip fast kernel if mode is "flash"
        if ENABLE_FAST_NORM and triton_rmsnorm_forward is not None and self.mode != "flash":
            if attn.norm_q is not None:
                query = triton_rmsnorm_forward(query, attn.norm_q.weight, attn.norm_q.eps)
            if attn.norm_k is not None:
                key = triton_rmsnorm_forward(key, attn.norm_k.weight, attn.norm_k.eps)
        else:
            # Fallback to PyTorch implementation
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

        # Reshape to (B, H, L, D) format after normalization.
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()

        if rotary_emb is not None:
            # Preprocess rotary_emb for the fast CUDA kernel.
            # Convert complex tensor to tuple (freqs_real, freqs_imag) for fast kernel
            # Skip fast kernel if mode is "flash"
            if ENABLE_FAST_ROPE and apply_rotary_emb_fast is not None and self.mode != "flash":
                # Preprocess: convert complex tensor to tuple format
                if isinstance(rotary_emb, torch.Tensor) and rotary_emb.is_complex():
                    # Convert complex frequencies to explicit real/imag tensors.
                    rot_real = rotary_emb.real.squeeze(0).squeeze(0).contiguous().to(torch.float32)
                    rot_imag = rotary_emb.imag.squeeze(0).squeeze(0).contiguous().to(torch.float32)
                    rotary_emb = (rot_real, rot_imag)
                
                # Now rotary_emb is in tuple format, use fast kernel
                try:
                    query, key = apply_rotary_emb_fast(query, key, rotary_emb)
                except (ValueError, TypeError) as e:
                    # If fast kernel fails, fall back to PyTorch
                    logger.warning(f"Fast RoPE kernel failed: {e}. Falling back to PyTorch implementation.")
                    # Restore original rotary_emb for PyTorch fallback
                    if isinstance(rotary_emb, tuple):
                        # Reconstruct complex tensor from tuple for PyTorch
                        rotary_emb = torch.complex(rotary_emb[0], rotary_emb[1])
                    
                    def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                        dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
                        x_rotated = torch.view_as_complex(hidden_states.to(dtype).unflatten(3, (-1, 2)))
                        x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                        return x_out.type_as(hidden_states)

                    query = apply_rotary_emb(query, rotary_emb)
                    key = apply_rotary_emb(key, rotary_emb)
            else:
                # Fallback to PyTorch implementation (rotary_emb is complex tensor)
                def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                    dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
                    x_rotated = torch.view_as_complex(hidden_states.to(dtype).unflatten(3, (-1, 2)))
                    x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                    return x_out.type_as(hidden_states)

                query = apply_rotary_emb(query, rotary_emb)
                key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            # Apply normalization before transpose.
            # Skip fast kernel if mode is "flash"
            if ENABLE_FAST_NORM and triton_rmsnorm_forward is not None and attn.norm_added_k is not None and self.mode != "flash":
                # Normalize on (B, L, H*D) shape, triton_rmsnorm_forward handles reshaping internally
                key_img = triton_rmsnorm_forward(key_img, attn.norm_added_k.weight, attn.norm_added_k.eps)
            elif attn.norm_added_k is not None:
                # Fallback to PyTorch implementation
                key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
            
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                hidden_states_img = F.scaled_dot_product_attention(
                    query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
                )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        B, H, S_q, D = query.shape
        B, H, S_k, D = key.shape
        s_k = attention_mask.sum().item() if attention_mask else S_k
        cu_seqlens_q = torch.tensor([0, S_q, S_q], dtype=torch.int32, device=query.device)
        cu_seqlens_kv = torch.tensor([0, s_k, S_k], dtype=torch.int32, device=query.device)
        max_seqlen_q, max_seqlen_kv = S_q, S_k
        
        if self.mode == "dfs" and self.cross_flag is False:
            if self.layer_idx not in self.skip_layers and self.step_idx >= self.skip_steps * 2: 
                hidden_states = dfs_attention(
                    query,
                    key,
                    value,
                    step_idx=self.step_idx,
                    skip_steps=self.skip_steps,
                    cache_interval=self.cache_interval,
                    layer_idx=self.layer_idx,
                    sparsity=self.sparsity,
                    sparsity_dcrt=self.sparsity_dcrt,
                    tile_size=self.tile_size,
                    block_size=self.block_size,
                    video_perm=self.video_perm,
                    cache_flag=self.cache_flag,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                )
            else:
                hidden_states = full_attention(
                    query, key, value, 
                    mode="flash", drop_rate=0.0, attn_mask=attention_mask, causal=False, \
                    cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, \
                    max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv, batch_size=query.shape[0])
            
        else:
            hidden_states = full_attention(
                    query, key, value, 
                    mode="flash", drop_rate=0.0, attn_mask=attention_mask, causal=False, \
                    cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, \
                    max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv, batch_size=query.shape[0])


        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        self.step_idx += 1
        # Reset after num_inference_steps (default 50)
        if self.step_idx == 50 * 2:
            self.step_idx = 0
        return hidden_states
