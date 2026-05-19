import torch
import numpy as np
import math
from hilbert import decode, encode



def gilbert2d_fast(width, height):
    """
    A fully iterative version of the Gilbert space-filling curve generator.
    Produces exactly the same order as the recursive version.
    """
    result = []
    stack = []

    # initial call
    if width >= height:
        stack.append((0, 0, width, 0, 0, height))
    else:
        stack.append((0, 0, 0, height, width, 0))

    def sgn(x):
        return -1 if x < 0 else (1 if x > 0 else 0)

    while stack:
        x, y, ax, ay, bx, by = stack.pop()

        w = abs(ax + ay)
        h = abs(bx + by)
        dax, day = sgn(ax), sgn(ay)
        dbx, dby = sgn(bx), sgn(by)

        # base case: line
        if h == 1 or w == 1:
            if h == 1:
                for _ in range(w):
                    result.append((x, y))
                    x, y = x + dax, y + day
            else:
                for _ in range(h):
                    result.append((x, y))
                    x, y = x + dbx, y + dby
            continue

        # subdivide
        ax2, ay2 = ax // 2, ay // 2
        bx2, by2 = bx // 2, by // 2
        w2 = abs(ax2 + ay2)
        h2 = abs(bx2 + by2)

        if 2 * w > 3 * h:
            if w2 % 2 and w > 2:
                ax2, ay2 = ax2 + dax, ay2 + day

            # recursive order (reverse push order)
            stack.append((x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by))
            stack.append((x, y, ax2, ay2, bx, by))

        else:
            if h2 % 2 and h > 2:
                bx2, by2 = bx2 + dbx, by2 + dby

            # recursive order (reverse push order)
            stack.append((
                x + (ax - dax) + (bx2 - dbx),
                y + (ay - day) + (by2 - dby),
                -bx2, -by2, -(ax - ax2), -(ay - ay2)
            ))
            stack.append((x + bx2, y + by2, ax, ay, bx - bx2, by - by2))
            stack.append((x, y, bx2, by2, ax2, ay2))

    return result


def hilbert2d_perm(f: int, h: int, w: int, device: str = 'cpu') -> torch.Tensor:
    """
    Fully identical output as your original hilbert_video_perm(),
    but MUCH faster (no recursion).
    """

    # 1. generate exact same (x,y) order
    path = gilbert2d_fast(w, h)

    # 2. convert to frame index
    frame_perm = torch.tensor([y * w + x for (x, y) in path], dtype=torch.long, device=device)

    # 3. expand to all frames
    base = torch.arange(f, dtype=torch.long, device=device)[:, None] * (h * w)
    return (frame_perm[None, :] + base).reshape(-1)


# ---------- bit-interleave (non in-place, vectorized) ----------
def part1by2_tensor(n: torch.Tensor) -> torch.Tensor:
    """
    Spread the bits of each 64-bit integer so that there are 2 zeros between each original bit.
    Implemented in a non-inplace, vectorized way. Expects dtype=torch.long.
    """
    # ensure long dtype
    n = n.to(torch.long)

    n = n & 0x1fffff  # keep lower 21 bits (safe for typical dims)
    n = (n | (n << 32)) & 0x1f00000000ffff
    n = (n | (n << 16)) & 0x1f0000ff0000ff
    n = (n | (n << 8))  & 0x100f00f00f00f00f
    n = (n | (n << 4))  & 0x10c30c30c30c30c3
    n = (n | (n << 2))  & 0x1249249249249249
    return n

def part1by1_tensor(n: torch.Tensor) -> torch.Tensor:
    """
    Spread the bits of each 64-bit integer so that there is 1 zero between each original bit.
    Used for 2D Morton codes. Expects dtype=torch.long.
    """
    n = n.to(torch.long)
    
    n = n & 0xffffffff  # keep lower 32 bits (safe for typical dims)
    n = (n | (n << 16)) & 0x0000FFFF0000FFFF
    n = (n | (n << 8))  & 0x00FF00FF00FF00FF
    n = (n | (n << 4))  & 0x0F0F0F0F0F0F0F0F
    n = (n | (n << 2))  & 0x3333333333333333
    n = (n | (n << 1))  & 0x5555555555555555
    return n

def morton3D_tensor(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Compute 3D Morton code for vectors x, y, z (all 1D tensors same length).
    Returns a 1D tensor of Morton codes (dtype long).
    """
    return part1by2_tensor(x) | (part1by2_tensor(y) << 1) | (part1by2_tensor(z) << 2)

def morton2D_tensor(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute 2D Morton code for vectors x, y (all 1D tensors same length).
    Returns a 1D tensor of Morton codes (dtype long).
    """
    return part1by1_tensor(x) | (part1by1_tensor(y) << 1)

# ---------- fast Morton permutation generator ----------
def morton2d_perm(f: int, h: int, w: int, device='cpu') -> torch.Tensor:
    """
    Return a 1D torch.LongTensor of length (f*h*w) with linear indices
    [0..f*h*w-1] arranged in 2D Morton (Z-order) within each frame,
    keeping frame dimension unchanged.

    The linear index ordering assumed is: idx = z*(h*w) + y*w + x
    Permutation only happens in spatial dimensions (h, w) within each frame.
    """
    dtype = torch.long
    
    # Generate coordinates for a single frame
    xs = torch.arange(w, dtype=dtype, device=device)
    ys = torch.arange(h, dtype=dtype, device=device)
    
    # Create all (x, y) pairs for one frame
    # x varies fastest, then y (so linear_idx = y*w + x)
    X = xs.repeat_interleave(h)                            # length = w*h
    Y = ys.repeat(w)                                       # length = w*h
    
    # Compute 2D Morton codes for spatial coordinates
    morton_codes = morton2D_tensor(X, Y)
    
    # Linear indices within a single frame
    frame_linear_idx = (Y * w + X).to(dtype)
    
    # Sort by Morton codes to get permutation within frame
    order = torch.argsort(morton_codes)
    frame_perm = frame_linear_idx[order]  # shape = (h*w,)
    
    # Expand to all frames: add base offset for each frame
    base = torch.arange(f, dtype=dtype, device=device)[:, None] * (h * w)
    perm = (frame_perm[None, :] + base).reshape(-1)
    
    return perm

def morton3d_perm(f: int, h: int, w: int, device='cpu') -> torch.Tensor:
    """
    Return a 1D torch.LongTensor of length (f*h*w) with linear indices
    [0..f*h*w-1] arranged in Morton (Z-order) order.

    The linear index ordering assumed is: idx = z*(h*w) + y*w + x
    """
    dtype = torch.long
    # create flattened coordinates (no meshgrid view / no ambiguous in-place)
    # produce x fastest (like standard row-major with x as inner dim)
    # shapes: total = w*h*f
    xs = torch.arange(w, dtype=dtype, device=device)
    ys = torch.arange(h, dtype=dtype, device=device)
    zs = torch.arange(f, dtype=dtype, device=device)

    # Expand to flattened coordinate lists
    # x varies fastest, then y, then z (so linear_idx = z*(h*w) + y*w + x)
    X = xs.repeat_interleave(h * f)                        # length = w*h*f
    Y = ys.repeat_interleave(f).repeat(w)                  # length = w*h*f
    Z = zs.repeat(w * h)                                   # length = w*h*f

    # Compute Morton codes and corresponding linear indices
    morton_codes = morton3D_tensor(X, Y, Z)
    linear_idx = (Z * (h * w) + Y * w + X).to(dtype)

    # argsort morton codes -> order of positions in Morton curve
    order = torch.argsort(morton_codes)
    perm = linear_idx[order]

    return perm

def hilbert3d_perm(f: int, h: int, w: int, device='cpu') -> torch.Tensor:
    """
    Return a 1D torch.LongTensor of length (f*h*w) with linear indices
    [0..f*h*w-1] arranged in 3D Hilbert curve order.

    The linear index ordering assumed is: idx = z*(h*w) + y*w + x
    Uses the hilbert library to encode coordinates and sort by Hilbert order.
    
    Args:
        f: Number of frames
        h: Height
        w: Width
        device: Device for the tensor
        
    Returns:
        Permutation tensor of shape (f*h*w,). When applied as tensor[perm],
        rearranges the video following the 3D Hilbert space-filling curve.
    """
    dtype = torch.long
    num_dims = 3
    
    # Calculate num_bits adaptively based on the maximum dimension
    max_dim = max(w, h, f)
    num_bits = max(1, math.ceil(math.log2(max_dim))) if max_dim > 1 else 1
    
    # Generate all coordinate combinations
    xs = torch.arange(w, dtype=dtype, device=device)
    ys = torch.arange(h, dtype=dtype, device=device)
    zs = torch.arange(f, dtype=dtype, device=device)

    # Expand to flattened coordinate lists
    # x varies fastest, then y, then z (so linear_idx = z*(h*w) + y*w + x)
    X = xs.repeat_interleave(h * f)                        # length = w*h*f
    Y = ys.repeat_interleave(f).repeat(w)                  # length = w*h*f
    Z = zs.repeat(w * h)                                   # length = w*h*f

    # Calculate original linear indices (f,h,w order)
    linear_idx = (Z * (h * w) + Y * w + X).to(dtype)

    # Convert to numpy for hilbert library (coordinates as [x, y, z] format)
    # Note: hilbert library expects coordinates in the format [x, y, z]
    coords_np = np.column_stack([X.cpu().numpy(), Y.cpu().numpy(), Z.cpu().numpy()])
    
    # Encode coordinates to Hilbert integers
    hilbert_codes = encode(coords_np, num_dims, num_bits)
    
    # Convert back to torch tensor and sort by Hilbert codes
    # Convert to int64 to avoid uint64 dtype issues with PyTorch sorting
    hilbert_codes_tensor = torch.from_numpy(hilbert_codes.astype(np.int64)).to(device, dtype=dtype)
    order = torch.argsort(hilbert_codes_tensor)
    perm = linear_idx[order]

    return perm

def hwf(f: int, h: int, w: int, device='cpu') -> torch.Tensor:
    """
    Return a 1D torch.LongTensor of length (f*h*w) that permutes a video
    from (f,h,w) flattened order to (h,w,f) flattened order.
    
    Original order (f,h,w): idx = z*(h*w) + y*w + x
      - All pixels in frame 0, then all pixels in frame 1, etc.
    
    Target order (h,w,f): idx = y*(w*f) + x*f + z
      - For each spatial position (y,x), all frames are consecutive
    
    Args:
        f: Number of frames
        h: Height
        w: Width
        device: Device for the tensor
        
    Returns:
        Permutation tensor of shape (f*h*w,). When applied as tensor[perm],
        converts from (f,h,w) order to (h,w,f) order.
    """
    dtype = torch.long
    total = f * h * w
    
    # Generate all coordinate combinations
    xs = torch.arange(w, dtype=dtype, device=device)
    ys = torch.arange(h, dtype=dtype, device=device)
    zs = torch.arange(f, dtype=dtype, device=device)
    
    # Expand to flattened coordinate lists (original order: z varies slowest)
    # x varies fastest, then y, then z (so linear_idx = z*(h*w) + y*w + x)
    X = xs.repeat_interleave(h * f)                        # length = w*h*f
    Y = ys.repeat_interleave(f).repeat(w)                  # length = w*h*f
    Z = zs.repeat(w * h)                                   # length = w*h*f
    
    # Calculate original indices (f,h,w order)
    orig_idx = (Z * (h * w) + Y * w + X).to(dtype)
    
    # Calculate target indices (h,w,f order)
    # For (h,w,f): y varies slowest, then x, then z (fastest)
    target_idx = (Y * (w * f) + X * f + Z).to(dtype)
    
    # Create permutation: perm[target_idx] = orig_idx
    # This means: output[target_idx] = input[orig_idx]
    perm = torch.zeros(total, dtype=dtype, device=device)
    perm[target_idx] = orig_idx
    
    return perm

def fwh(f: int, h: int, w: int, device='cpu') -> torch.Tensor:
    """
    Return a 1D torch.LongTensor of length (f*h*w) that permutes a video
    from (f,h,w) flattened order to (f,w,h) flattened order.
    
    Original order (f,h,w): idx = z*(h*w) + y*w + x
      - All pixels in frame 0, then all pixels in frame 1, etc.
    
    Target order (f,w,h): idx = z*(w*h) + x*h + y
      - For each frame z, all pixels are ordered by width then height.
    
    Args:
        f: Number of frames
        h: Height
        w: Width
        device: Device for the tensor
        
    Returns:
        Permutation tensor of shape (f*h*w,). When applied as tensor[perm],
        converts from (f,h,w) order to (f,w,h) order.
    """
    dtype = torch.long
    total = f * h * w

    # Generate all coordinate combinations
    xs = torch.arange(w, dtype=dtype, device=device)
    ys = torch.arange(h, dtype=dtype, device=device)
    zs = torch.arange(f, dtype=dtype, device=device)

    # Expand to flattened coordinate lists (original order: z varies slowest)
    # x varies fastest, then y, then z (so linear_idx = z*(h*w) + y*w + x)
    X = xs.repeat_interleave(h * f)                        # length = w*h*f
    Y = ys.repeat_interleave(f).repeat(w)                  # length = w*h*f
    Z = zs.repeat(w * h)                                   # length = w*h*f

    # Calculate original indices (f,h,w order)
    orig_idx = (Z * (h * w) + Y * w + X).to(dtype)

    # Calculate target indices (f,w,h order)
    # For (f,w,h): z varies slowest, then x, then y (fastest)
    target_idx = (Z * (w * h) + X * h + Y).to(dtype)

    # Create permutation: perm[target_idx] = orig_idx
    # This means: output[target_idx] = input[orig_idx]
    perm = torch.zeros(total, dtype=dtype, device=device)
    perm[target_idx] = orig_idx
    return perm

def block3d_perm(f: int, h: int, w: int, a: int, b: int, c: int, device='cpu') -> torch.Tensor:
    """
    Return a 1D torch.LongTensor of length (f*h*w) that reorganizes a 3D sequence
    into blocks of size (a, b, c) in (f, h, w) dimensions.
    
    The video is divided into blocks where:
      - Block size in frame dimension: a
      - Block size in height dimension: b
      - Block size in width dimension: c
    
    Ordering: Blocks are ordered first (by block_z, block_y, block_x),
    then within each block, elements are ordered by (local_z, local_y, local_x).
    
    Original order (f,h,w): idx = z*(h*w) + y*w + x
      - All pixels in frame 0, then all pixels in frame 1, etc.
    
    Block order: Elements are grouped into blocks and ordered block-by-block,
    with within-block ordering following (local_z, local_y, local_x).
    
    Args:
        f: Number of frames
        h: Height
        w: Width
        a: Block size in frame dimension
        b: Block size in height dimension
        c: Block size in width dimension
        device: Device for the tensor
        
    Returns:
        Permutation tensor of shape (f*h*w,). When applied as tensor[perm],
        reorganizes the video into blocks of size (a, b, c).
    """
    dtype = torch.long
    
    # Generate all coordinate combinations
    xs = torch.arange(w, dtype=dtype, device=device)
    ys = torch.arange(h, dtype=dtype, device=device)
    zs = torch.arange(f, dtype=dtype, device=device)
    
    # Expand to flattened coordinate lists (original order: z varies slowest)
    # x varies fastest, then y, then z (so linear_idx = z*(h*w) + y*w + x)
    X = xs.repeat_interleave(h * f)                        # length = w*h*f
    Y = ys.repeat_interleave(f).repeat(w)                  # length = w*h*f
    Z = zs.repeat(w * h)                                   # length = w*h*f
    
    # Calculate original indices (f,h,w order)
    orig_idx = (Z * (h * w) + Y * w + X).to(dtype)
    
    # Calculate block coordinates
    block_z = Z // a
    block_y = Y // b
    block_x = X // c
    
    # Calculate local coordinates within block
    local_z = Z % a
    local_y = Y % b
    local_x = X % c
    
    # Calculate number of blocks in each dimension
    num_blocks_z = (f + a - 1) // a  # ceil division
    num_blocks_y = (h + b - 1) // b
    num_blocks_x = (w + c - 1) // c
    
    # Create sorting key: block order first, then within-block order
    # Block key: block_z * (num_blocks_y * num_blocks_x) + block_y * num_blocks_x + block_x
    block_key = block_z * (num_blocks_y * num_blocks_x) + block_y * num_blocks_x + block_x
    
    # Within-block key: local_z * (b * c) + local_y * c + local_x
    within_block_key = local_z * (b * c) + local_y * c + local_x
    
    # Combined key: prioritize block order, then within-block order
    # Use a large multiplier to ensure block order dominates
    max_within_block = a * b * c
    sort_key = block_key * max_within_block + within_block_key
    
    # Sort by key to get permutation
    order = torch.argsort(sort_key)
    perm = orig_idx[order]
    
    return perm
