import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention
from typing import Optional

from .fullattention import full_attention
from .attention_hyvideo import dfs_attention
from dfsattn.utils.logger import logger

# Try to import fast kernels
try:
    from .kernels import ENABLE_FAST_KERNEL, apply_qk_norm, apply_qk_rope_single, apply_qk_rope_double
    if ENABLE_FAST_KERNEL:
        logger.info("Fast CUDA/Triton kernels enabled for QK normalization and RoPE")
except ImportError:
    ENABLE_FAST_KERNEL = False
    apply_qk_norm = None
    apply_qk_rope_single = None
    apply_qk_rope_double = None
    logger.info("Fast CUDA/Triton kernels not available, using PyTorch fallback")


class HunyuanVideo_DFSAttn_Processor2_0:
    def __init__(
        self,
        mode,
        sparsity,
        tile_size,
        block_size,
        video_len,
        video_perm,
        processor_id,
        skip_layers,
        skip_steps,
        cache_interval,
        sparsity_dcrt,
        cache_flag,
    ):
        self.mode = mode
        self.sparsity = sparsity
        self.tile_size = tile_size
        self.block_size = block_size
        self.video_len = video_len
        self.video_perm = video_perm
        self.step_idx = 0
        self.layer_idx = processor_id
        self.skip_layers = skip_layers
        self.skip_steps = skip_steps
        self.cache_interval = cache_interval
        self.sparsity_dcrt = sparsity_dcrt
        self.cache_flag = cache_flag

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("HunyuanVideoAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")
    
    def get_cu_max_seqlen(self, attention_mask, device):
        cu_seqlens_q = torch.tensor([0, attention_mask.sum(), attention_mask.numel()], dtype=torch.int32, device=device)
        cu_seqlens_kv = torch.tensor(
            [0, attention_mask.sum(), attention_mask.numel()], dtype=torch.int32, device=device
        )
        max_seqlen_q = attention_mask.numel()
        max_seqlen_kv = attention_mask.numel()
        return cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()

        # 2. QK normalization - use fast kernel if available (skip if mode is "flash")
        if ENABLE_FAST_KERNEL and apply_qk_norm is not None and self.mode != "flash":
            query, key = apply_qk_norm(attn.norm_q, attn.norm_k, query, key)
        else:
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream - use fast kernel if available (skip if mode is "flash")
        if image_rotary_emb is not None:
            if ENABLE_FAST_KERNEL and apply_qk_rope_single is not None and apply_qk_rope_double is not None and self.mode != "flash":
                if attn.add_q_proj is None and encoder_hidden_states is not None:
                    # Single stream: text after video
                    query, key = apply_qk_rope_single(query, key, image_rotary_emb, encoder_hidden_states)
                else:
                    # Double stream: no text
                    query, key = apply_qk_rope_double(query, key, image_rotary_emb)
            else:
                # Fallback to PyTorch implementation
                from diffusers.models.embeddings import apply_rotary_emb

                if attn.add_q_proj is None and encoder_hidden_states is not None:
                    query = torch.cat(
                        [
                            apply_rotary_emb(query[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                            query[:, :, -encoder_hidden_states.shape[1] :],
                        ],
                        dim=2,
                    )
                    key = torch.cat(
                        [
                            apply_rotary_emb(key[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                            key[:, :, -encoder_hidden_states.shape[1] :],
                        ],
                        dim=2,
                    )
                else:
                    query = apply_rotary_emb(query, image_rotary_emb)
                    key = apply_rotary_emb(key, image_rotary_emb)

        # 4. Encoder condition QKV projection and normalization
        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2) # [B, H, L, D]
            key = torch.cat([key, encoder_key], dim=2) # [B, H, L, D]
            value = torch.cat([value, encoder_value], dim=2) # [B, H, L, D]

        # 5. Attention
        cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv = self.get_cu_max_seqlen(attention_mask, query.device)
        if self.mode == "dfs":
            if self.layer_idx not in self.skip_layers and self.step_idx >= self.skip_steps: 
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
                    video_len=self.video_len,
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
            
        elif self.mode in ["flash", "torch", "vanilla"]:
            hidden_states = full_attention(
                    query, key, value, 
                    mode=self.mode, drop_rate=0.0, attn_mask=attention_mask, causal=False, \
                    cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, \
                    max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv, batch_size=query.shape[0])
        
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)
        
        # 6. Output projection
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        self.step_idx += 1
        if self.step_idx == 50:
            self.step_idx = 0

        return hidden_states, encoder_hidden_states
