"""
Fast CUDA/Triton kernels for optimized attention operations.
Provides fast implementations of RMSNorm and RoPE operations.
"""

try:
    import importlib.util
    import os
    import sys

    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(kernel_dir, "build")

    if build_dir not in sys.path:
        sys.path.insert(0, build_dir)

    try:
        import _kernels
    except ImportError:
        if not os.path.isdir(build_dir):
            raise ImportError(f"Kernel build directory not found: {build_dir}")

        so_files = [f for f in os.listdir(build_dir) if f.startswith("_kernels") and f.endswith(".so")]
        if not so_files:
            raise ImportError(f"No _kernels*.so file found in {build_dir}")

        so_path = os.path.join(build_dir, so_files[0])
        spec = importlib.util.spec_from_file_location("_kernels", so_path)
        _kernels = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_kernels)

    ENABLE_FAST_KERNEL = True
    
    def apply_qk_norm(attn_norm_q, attn_norm_k, query, key):
        """
        Fast RMSNorm implementation using CUDA kernels.
        Input: query, key with shape (B, H, L, D)
        """
        assert query.is_contiguous() and key.is_contiguous(), "Query and Key must be contiguous"
        
        # Reshape to (B*H*L, D) for kernel
        query_flat = query.view(-1, query.shape[-1])
        key_flat = key.view(-1, key.shape[-1])
        
        # Apply RMSNorm in-place
        if attn_norm_q is not None:
            _kernels.rms_norm_forward(query_flat, attn_norm_q.weight, attn_norm_q.eps)
        if attn_norm_k is not None:
            _kernels.rms_norm_forward(key_flat, attn_norm_k.weight, attn_norm_k.eps)
        
        return query, key
    
    def apply_qk_rope_single(query, key, image_rotary_emb, encoder_hidden_states):
        """
        Fast RoPE for single stream (text after video).
        Input: query, key with shape (B, H, L, D)
        """
        assert query.is_contiguous() and key.is_contiguous(), "Query and Key must be contiguous"
        
        txt_len = encoder_hidden_states.shape[1]
        cos, sin = image_rotary_emb[0], image_rotary_emb[1]
        _kernels.apply_qk_rope_inplace_cossin_txtlast(query, key, cos, sin, txt_len)
        
        return query, key
    
    def apply_qk_rope_double(query, key, image_rotary_emb):
        """
        Fast RoPE for double stream (no text).
        Input: query, key with shape (B, H, L, D)
        """
        assert query.is_contiguous() and key.is_contiguous(), "Query and Key must be contiguous"
        
        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb[0], image_rotary_emb[1]
            _kernels.apply_qk_rope_inplace_cossin_txtlast(query, key, cos, sin, 0)
        
        return query, key

except ImportError:
    import torch
    ENABLE_FAST_KERNEL = False
    
    def apply_qk_norm(attn_norm_q, attn_norm_k, query, key):
        """Fallback to PyTorch implementation"""
        if attn_norm_q is not None:
            query = attn_norm_q(query)
        if attn_norm_k is not None:
            key = attn_norm_k(key)
        return query, key
    
    def apply_qk_rope_single(query, key, image_rotary_emb, encoder_hidden_states):
        """Fallback to PyTorch implementation"""
        from diffusers.models.embeddings import apply_rotary_emb
        
        txt_len = encoder_hidden_states.shape[1]
        img_q, txt_q = query[:, :, :-txt_len], query[:, :, -txt_len:]
        img_k, txt_k = key[:, :, :-txt_len], key[:, :, -txt_len:]
        
        img_q = apply_rotary_emb(img_q, image_rotary_emb)
        img_k = apply_rotary_emb(img_k, image_rotary_emb)
        
        query = torch.cat([img_q, txt_q], dim=2)
        key = torch.cat([img_k, txt_k], dim=2)
        
        return query, key
    
    def apply_qk_rope_double(query, key, image_rotary_emb):
        """Fallback to PyTorch implementation"""
        from diffusers.models.embeddings import apply_rotary_emb
        
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        
        return query, key
