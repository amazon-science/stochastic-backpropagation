import torch
import random
import numpy as np

def mask2d_to_index(mask):
    H, W = mask.shape
    mask.view(-1)
    i, j = np.indices((H, W))
    all_idx = i * W + j
    keep_idx, drop_idx = all_idx[mask], all_idx[~mask]
    return list(keep_idx), list(drop_idx)

def index_to_mask2d(keep_idx, H, W):
    mask = torch.zeros(H * W)
    mask[keep_idx] = 1
    mask = mask.view(H, W).to(torch.bool)
    return mask

def generate_grid_mask2d(H, W, keep_ratio, gird_start_point):
    assert(keep_ratio <= 1.0)
    if keep_ratio <= 0.5:
        grid_stride = int(np.round(1.0 / keep_ratio))
        gird_start_point %= grid_stride
        keep_mask = np.indices((H, W)).sum(0) % grid_stride == gird_start_point
    else:
        drop_ratio = 1 - keep_ratio
        grid_stride = int(np.round(1.0 / drop_ratio))
        gird_start_point %= grid_stride
        keep_mask = np.indices((H, W)).sum(0) % grid_stride != gird_start_point
    keep_mask = torch.from_numpy(keep_mask)
    keep_idx, drop_idx = mask2d_to_index(keep_mask)
    return keep_mask, keep_idx, drop_idx

def generate_random_mask2d(H, W, keep_ratio, base_keep_idx=[]):
    drop_ratio = 1 - keep_ratio
    total_size = H * W
    all_candidates = set(range(total_size))
    
    # exclude the base_keep_idx:
    #    if we have a base_keep_idx in the early layers, then we have to keep them for later layers as well
    drop_candidates = all_candidates.difference(set(base_keep_idx))
    drop_size = int(drop_ratio * total_size)
    assert len(drop_candidates) >= drop_size, 'the keep_ratio has to be larger than the base_keep_ratio'
    
    drop_idx = random.sample(drop_candidates, drop_size)
    keep_idx = all_candidates.difference(set(drop_idx))
    
    keep_idx, drop_idx = list(keep_idx), list(drop_idx)
    keep_mask = index_to_mask2d(keep_idx, H, W)
    
    return keep_mask, keep_idx, drop_idx

def generate_mask2d_with_base(H, W, block_size, base_mask):
    bH, bW = base_mask.shape
    assert(bH * block_size == H and bW * block_size == W)
    h = H // block_size
    w = W // block_size
    keep_mask = np.zeros((H, W), dtype=np.float32)
    for j in range(h):
        for k in range(w):
            keep_mask[
                j * block_size : (j + 1) * block_size,
                k * block_size : (k + 1) * block_size,
            ] = base_mask[j, k]
    keep_mask = torch.from_numpy(keep_mask).to(torch.bool)
    keep_idx, drop_idx = mask2d_to_index(keep_mask)
    return keep_mask, keep_idx, drop_idx
