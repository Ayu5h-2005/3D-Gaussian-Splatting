"""
tile_invalidation.py - Screen-Space Tile Invalidation & Verification Engine

Computes conservative screen-space invalidation tiles using the union of
each modified 3D Gaussian's old (t-1) and new (t) projected bounding footprints.
Ensures zero false negatives while maximizing tile reuse in unchanged regions.
"""

import math
from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F
import gsplat


def compute_stats_from_mask(
    invalid_tile_mask: torch.Tensor,
    tile_size: int,
    width: int,
    height: int,
) -> Dict[str, Union[int, float, Tuple[int, int]]]:
    """Computes scalar tile invalidation metrics from a 2D boolean mask."""
    tile_h, tile_w = invalid_tile_mask.shape
    total_tiles = tile_h * tile_w
    invalid_count = int(invalid_tile_mask.sum().item())
    clean_count = total_tiles - invalid_count
    return {
        "tile_size": tile_size,
        "grid_resolution": (tile_h, tile_w),
        "total_tiles": total_tiles,
        "invalid_tile_count": invalid_count,
        "clean_tile_count": clean_count,
        "invalid_tile_percentage": (invalid_count / total_tiles) * 100.0 if total_tiles > 0 else 0.0,
        "clean_tile_percentage": (clean_count / total_tiles) * 100.0 if total_tiles > 0 else 100.0,
    }


def compute_tile_invalidation_mask(
    scene_old: Dict[str, Union[torch.Tensor, int]],
    scene_new: Dict[str, Union[torch.Tensor, int]],
    changed_indices: torch.Tensor,
    tile_size: int = 16,
    compute_stats: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Computes a conservative 2D boolean tile invalidation mask representing
    all screen tiles affected by the movement/modification of target Gaussians.

    The mask is formed by the union of projected bounding boxes at t-1 and t.
    Uses GPU-vectorized subset projection and tensor-core matrix marking
    with zero host-device synchronization when compute_stats=False.

    Args:
        scene_old: Scene dictionary at frame t-1.
        scene_new: Scene dictionary at frame t.
        changed_indices: 1D Tensor of Gaussian indices that were modified.
        tile_size: Screen-space tile dimension in pixels (default: 16).
        compute_stats: Whether to compute scalar statistics (requires .item() sync).
                       Set to False in benchmarking inner loops to avoid host stalls.

    Returns:
        invalid_tile_mask: 2D boolean Tensor of shape (tile_height, tile_width).
        invalid_pixel_mask: 3D boolean Tensor of shape (height, width, 1).
        invalidation_stats: Dictionary containing diagnostic tile metrics.
    """
    device = scene_old["means"].device
    width = int(scene_old["width"])
    height = int(scene_old["height"])

    tile_w = (width + tile_size - 1) // tile_size
    tile_h = (height + tile_size - 1) // tile_size

    if changed_indices.numel() == 0:
        # No change: all tiles are clean
        invalid_tile_mask = torch.zeros((tile_h, tile_w), dtype=torch.bool, device=device)
        invalid_pixel_mask = torch.zeros((height, width, 1), dtype=torch.bool, device=device)
        stats = {
            "tile_size": tile_size,
            "grid_resolution": (tile_h, tile_w),
            "total_tiles": tile_h * tile_w,
            "invalid_tile_count": 0,
            "clean_tile_count": tile_h * tile_w,
            "invalid_tile_percentage": 0.0,
            "clean_tile_percentage": 100.0,
        }
        return invalid_tile_mask, invalid_pixel_mask, stats

    # 1. Project ONLY changed Gaussians at t-1 and t to obtain 2D means and 3-sigma bounding radii
    radii_old, means2d_old, depths_old, _, _ = gsplat.fully_fused_projection(
        scene_old["means"][changed_indices], None,
        scene_old["quats"][changed_indices], scene_old["scales"][changed_indices],
        scene_old["viewmats"], scene_old["Ks"], width, height, packed=False
    )
    radii_new, means2d_new, depths_new, _, _ = gsplat.fully_fused_projection(
        scene_new["means"][changed_indices], None,
        scene_new["quats"][changed_indices], scene_new["scales"][changed_indices],
        scene_new["viewmats"], scene_new["Ks"], width, height, packed=False
    )

    # 2. Concatenate old and new states for unified GPU processing
    m2d_all = torch.cat([means2d_old[0], means2d_new[0]], dim=0)   # [2K, 2]
    rad_all = torch.cat([radii_old[0], radii_new[0]], dim=0)       # [2K, 2]
    dep_all = torch.cat([depths_old[0], depths_new[0]], dim=0)     # [2K]

    # Valid projection filter: positive depth and non-zero radius
    valid = (dep_all > 0.01) & (rad_all > 0).all(dim=-1)

    # 2D Bounding box calculation
    xy_min = m2d_all - rad_all  # [2K, 2]
    xy_max = m2d_all + rad_all  # [2K, 2]

    # Frustum culling filter: check if bbox intersects screen [0, width] x [0, height]
    in_screen = (xy_max >= 0).all(dim=-1) & (xy_min[:, 0] < width) & (xy_min[:, 1] < height)
    active = valid & in_screen

    # Map to tile coordinates and clamp to grid boundaries
    uv_min = torch.clamp((xy_min / tile_size).floor().long(), 0, tile_w - 1)
    uv_max = torch.clamp((xy_max / tile_size).floor().long(), 0, tile_w - 1)
    u_min, v_min = uv_min[:, 0], uv_min[:, 1]
    u_max, v_max = uv_max[:, 0], uv_max[:, 1]

    # 3. Vectorized GPU Tile Marking (Zero CPU loops, Zero host-device synchronization)
    y_grid = torch.arange(tile_h, device=device).view(tile_h, 1)
    x_grid = torch.arange(tile_w, device=device).view(1, tile_w)

    Y = ((y_grid >= v_min.view(1, -1)) & (y_grid <= v_max.view(1, -1))).float() * active.view(1, -1).float()
    X = ((x_grid >= u_min.view(-1, 1)) & (x_grid <= u_max.view(-1, 1))).float()

    invalid_tile_mask = (torch.mm(Y, X) > 0)

    # 4. Upsample tile mask to screen resolution (exact 16x16 blocks)
    invalid_pixel_mask = (
        invalid_tile_mask.repeat_interleave(tile_size, dim=0)
        .repeat_interleave(tile_size, dim=1)[:height, :width, None]
    )

    if compute_stats:
        stats = compute_stats_from_mask(invalid_tile_mask, tile_size, width, height)
    else:
        stats = {
            "tile_size": tile_size,
            "grid_resolution": (tile_h, tile_w),
            "total_tiles": tile_h * tile_w,
            "invalid_tile_count": None,
            "clean_tile_count": None,
            "invalid_tile_percentage": None,
            "clean_tile_percentage": None,
        }

    return invalid_tile_mask, invalid_pixel_mask, stats


def compute_ground_truth_changed_tiles(
    img_old: torch.Tensor,
    img_new: torch.Tensor,
    tile_size: int = 16,
    threshold: float = 1.0 / 255.0,
) -> torch.Tensor:
    """
    Computes ground-truth changed tiles by detecting actual pixel differences
    between two full-scene renders. Fully GPU-vectorized via max-pooling.

    Args:
        img_old: Tensor of shape (H, W, C) in range [0, 1].
        img_new: Tensor of shape (H, W, C) in range [0, 1].
        tile_size: Tile dimension in pixels.
        threshold: Absolute difference threshold for non-trivial pixel change.

    Returns:
        gt_changed_tiles: 2D boolean Tensor of shape (tile_height, tile_width).
    """
    diff = (img_new - img_old).abs().max(dim=-1).values  # [H, W]
    changed_pixels = diff > threshold

    height, width = changed_pixels.shape
    tile_h = (height + tile_size - 1) // tile_size
    tile_w = (width + tile_size - 1) // tile_size

    pad_w = tile_w * tile_size - width
    pad_h = tile_h * tile_size - height
    if pad_w > 0 or pad_h > 0:
        padded = F.pad(changed_pixels.float()[None, None, ...], (0, pad_w, 0, pad_h))
    else:
        padded = changed_pixels.float()[None, None, ...]

    gt_tiles = F.max_pool2d(padded, kernel_size=tile_size, stride=tile_size)[0, 0] > 0
    return gt_tiles


def evaluate_invalidation_coverage(
    predicted_tiles: torch.Tensor,
    gt_tiles: torch.Tensor,
) -> Dict[str, Union[int, float, bool]]:
    """
    Verifies that predicted invalid tiles completely cover all ground-truth changed tiles
    (Zero False Negatives) and reports coverage metrics.
    """
    # False Negatives: Ground-truth changed tile NOT marked invalid (FATAL FLAW)
    false_negatives = (gt_tiles & (~predicted_tiles)).sum().item()

    # True Positives: Ground-truth changed tile correctly marked invalid
    true_positives = (gt_tiles & predicted_tiles).sum().item()

    # False Positives: Clean tile marked invalid (conservative safety margin)
    false_positives = ((~gt_tiles) & predicted_tiles).sum().item()

    # True Negatives: Clean tile correctly kept clean
    true_negatives = ((~gt_tiles) & (~predicted_tiles)).sum().item()

    total_gt_changed = int(gt_tiles.sum().item())
    recall = (true_positives / total_gt_changed) if total_gt_changed > 0 else 1.0
    precision = (true_positives / (true_positives + false_positives)) if (true_positives + false_positives) > 0 else 1.0

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "recall": recall,
        "precision": precision,
        "zero_false_negatives": (false_negatives == 0),
    }


def composite_with_cache(
    cached_image: torch.Tensor,
    sparse_rerender: torch.Tensor,
    mask: torch.Tensor,
    tile_size: int = 16,
) -> torch.Tensor:
    """
    Assembles the final frame by copying newly re-rendered pixels into invalid tiles
    and preserving the cached frame in all clean tiles.
    
    Supports:
    - 2D tile mask [tile_h, tile_w]: ultra-fast 5D broadcast (no allocation).
    - 3D pixel mask [H, W, 1]: direct torch.where (backward compatibility).
    """
    if cached_image.dim() == 4:
        cached_image = cached_image[0]
    if sparse_rerender.dim() == 4:
        sparse_rerender = sparse_rerender[0]

    if mask.dim() == 2:
        # Ultra-fast zero-allocation 5D broadcast
        th, tw = mask.shape
        h, w, c = cached_image.shape
        m5d = mask.view(th, 1, tw, 1, 1)
        c5d = cached_image.view(th, tile_size, tw, tile_size, c)
        s5d = sparse_rerender.view(th, tile_size, tw, tile_size, c)
        return torch.where(m5d, s5d, c5d).reshape(h, w, c)

    return torch.where(mask, sparse_rerender, cached_image)


def compute_image_metrics(
    reference: torch.Tensor,
    test: torch.Tensor,
    window_size: int = 11,
) -> Dict[str, float]:
    """
    Computes exact image similarity metrics between reference full render and test composite:
    PSNR, SSIM, Max Absolute Difference, L1 error, and MSE.
    """
    if reference.dim() == 4:
        reference = reference[0]
    if test.dim() == 4:
        test = test[0]

    diff = (reference - test).abs()
    max_err = diff.max().item()
    l1_err = diff.mean().item()
    mse = (diff ** 2).mean().item()

    psnr = 10.0 * math.log10(1.0 / mse) if mse > 1e-12 else float("inf")

    # Self-contained PyTorch GPU SSIM (Wang et al. 2004)
    ref_bchw = reference.permute(2, 0, 1)[None, ...]
    test_bchw = test.permute(2, 0, 1)[None, ...]
    channels = ref_bchw.size(1)

    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = g.outer(g)[None, None, ...].repeat(channels, 1, 1, 1).to(reference.device)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.conv2d(ref_bchw, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(test_bchw, window, padding=window_size // 2, groups=channels)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(ref_bchw * ref_bchw, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(test_bchw * test_bchw, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(ref_bchw * test_bchw, window, padding=window_size // 2, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_val = ssim_map.mean().item()

    return {
        "max_absolute_error": max_err,
        "l1_error": l1_err,
        "mse": mse,
        "psnr_db": psnr,
        "ssim": ssim_val,
    }
