import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

def attention_map_visualization(attn_maps, save_dir, step_idx, layer_idx, head_idx):

    step_dir = os.path.join(save_dir, f"step_{step_idx:03d}")
    layer_dir = os.path.join(step_dir, f"layer_{layer_idx:02d}")
    os.makedirs(layer_dir, exist_ok=True)
    attn_np = attn_maps.detach().cpu().numpy()

    N = attn_np.shape[1]
    threshold = 1 / max(N, 1)
    mask = attn_np <= threshold

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        attn_np,
        mask=mask,
        cmap='Reds',
        cbar=True,
        vmin=threshold,
        vmax=float(np.percentile(attn_np, 99.5)),
        square=True,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )
    ax.set_facecolor('white')
    ax.set_xlabel('Key')
    ax.set_ylabel('Query')
    ax.set_title(f'Attention Map - Step {step_idx}, Layer {layer_idx}, Head {head_idx}')

    # Grid and ticks per frame tokens (240) and black line for text region
    seq_len = attn_np.shape[0]
    text_len = 0
    frame_block = 240
    visual_len = max(seq_len - text_len, 0)

    # Minor grid every frame_block within visual region
    if visual_len > 0:
        minor_ticks = np.arange(0, visual_len + 1, frame_block)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(which='minor', color='#dddddd', linestyle='-', linewidth=0.4)

    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Save the visualization
    save_path = os.path.join(layer_dir, f"head_{head_idx:02d}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def mask_visualization(mask, save_dir, step_idx, layer_idx, head_idx, suffix=""):
    """
    Visualize a boolean mask as a heatmap.
    
    Args:
        mask (torch.Tensor): Boolean mask with shape [q_blocks, k_blocks] or similar 2D shape
        save_dir (str): Directory to save the visualization
        step_idx (int): Step index for naming
        layer_idx (int): Layer index for naming
        head_idx (int): Head index for naming
        suffix (str): Optional suffix for the filename
    """
    step_dir = os.path.join(save_dir, f"step_{step_idx:03d}")
    layer_dir = os.path.join(step_dir, f"layer_{layer_idx:02d}")
    os.makedirs(layer_dir, exist_ok=True)
    
    # Convert to numpy and handle boolean mask
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy().astype(float)
    else:
        mask_np = np.array(mask).astype(float)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        mask_np,
        cmap='Blues',
        cbar=True,
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )
    ax.set_facecolor('white')
    ax.set_xlabel('Key Blocks')
    ax.set_ylabel('Query Blocks')
    ax.set_title(f'Selected Mask - Step {step_idx}, Layer {layer_idx}, Head {head_idx}')
    
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    # Save the visualization
    filename = f"head_{head_idx:02d}_mask.png" if not suffix else f"head_{head_idx:02d}_mask_{suffix}.png"
    save_path = os.path.join(layer_dir, filename)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def attention_map_with_mask_visualization(
    attn_maps, 
    block_mask, 
    save_dir, 
    step_idx, 
    layer_idx, 
    head_idx, 
    block_size,
):
    """
    Visualize attention map with selected mask overlaid.
    
    Args:
        attn_maps (torch.Tensor): Attention map with shape [q_tokens, k_tokens] (token level)
        block_mask (torch.Tensor): Boolean mask with shape [q_blocks, k_blocks] (block level)
        save_dir (str): Directory to save the visualization
        step_idx (int): Step index for naming
        layer_idx (int): Layer index for naming
        head_idx (int): Head index for naming
        block_size (int): Size of each block in tokens (default: 60)
    """
    step_dir = os.path.join(save_dir, f"step_{step_idx:03d}")
    layer_dir = os.path.join(step_dir, f"layer_{layer_idx:02d}")
    os.makedirs(layer_dir, exist_ok=True)
    
    # Convert attention map to numpy
    attn_np = attn_maps.detach().cpu().numpy()
    q_tokens, k_tokens = attn_np.shape
    
    # Convert block mask to numpy
    if isinstance(block_mask, torch.Tensor):
        mask_np = block_mask.detach().cpu().numpy()
    else:
        mask_np = np.array(block_mask)
    
    q_blocks, k_blocks = mask_np.shape
    
    # Create token-level mask
    token_mask = np.zeros((q_tokens, k_tokens), dtype=bool)
    
    for q_block_idx in range(q_blocks):
        for k_block_idx in range(k_blocks):
            if mask_np[q_block_idx, k_block_idx]:
                # Calculate token ranges for this block
                q_start = q_block_idx * block_size
                q_end = min(q_start + block_size, q_tokens)
                k_start = k_block_idx * block_size
                k_end = min(k_start + block_size, k_tokens)
                
                # Mark this block region as selected
                token_mask[q_start:q_end, k_start:k_end] = True
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot attention map
    N = attn_np.shape[1]
    threshold = 1 / max(N, 1)
    attn_mask = attn_np <= threshold
    
    # First plot the attention heatmap
    sns.heatmap(
        attn_np,
        mask=attn_mask,
        cmap='Reds',
        cbar=True,
        vmin=threshold,
        vmax=float(np.percentile(attn_np, 99.5)),
        square=True,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )
    
    # Overlay the selected mask as block boundaries and colored regions
    # Draw rectangles around selected blocks
    from matplotlib.patches import Rectangle
    
    for q_block_idx in range(q_blocks):
        for k_block_idx in range(k_blocks):
            if mask_np[q_block_idx, k_block_idx]:
                # Calculate token ranges for this block
                q_start = q_block_idx * block_size
                q_end = min(q_start + block_size, q_tokens)
                k_start = k_block_idx * block_size
                k_end = min(k_start + block_size, k_tokens)
                
                # Draw rectangle outline for the block
                rect = Rectangle(
                    (k_start, q_start),
                    k_end - k_start,
                    q_end - q_start,
                    linewidth=2,
                    edgecolor='cyan',
                    facecolor='none',
                    alpha=0.8
                )
                ax.add_patch(rect)
    
    # Also add a subtle colored overlay for better visibility of selected regions
    overlay_alpha = np.zeros((q_tokens, k_tokens, 4))
    overlay_alpha[token_mask, 0] = 0.0  # R
    overlay_alpha[token_mask, 1] = 1.0  # G
    overlay_alpha[token_mask, 2] = 1.0  # B
    overlay_alpha[token_mask, 3] = 0.15  # Alpha (semi-transparent)
    
    # Match seaborn heatmap coordinate system (origin='upper')
    ax.imshow(overlay_alpha, aspect='auto', origin='upper', interpolation='nearest', extent=[0, k_tokens, 0, q_tokens])
    
    ax.set_facecolor('white')
    ax.set_xlabel('Key Tokens')
    ax.set_ylabel('Query Tokens')
    ax.set_title(f'Attention Map with Selected Mask - Step {step_idx}, Layer {layer_idx}, Head {head_idx}')
    
    # Grid and ticks
    seq_len = attn_np.shape[0]
    text_len = 0
    frame_block = 240
    visual_len = max(seq_len - text_len, 0)
    
    if visual_len > 0:
        minor_ticks = np.arange(0, visual_len + 1, frame_block)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(which='minor', color='#dddddd', linestyle='-', linewidth=0.4)
    
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    # Save the visualization
    save_path = os.path.join(layer_dir, f"head_{head_idx:02d}_with_mask.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)