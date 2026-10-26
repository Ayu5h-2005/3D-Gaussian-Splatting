import sys
import time
from pathlib import Path

ROOT = Path(r"D:\3dgs-incremental-research")
sys.path.insert(0, str(ROOT))

import torch
import gsplat
import numpy as np

from src.baseline_3dgs import generate_procedural_scene
from src.ply_io import save_3dgs_ply, load_3dgs_ply
from src.tile_invalidation import compute_image_metrics, compute_ground_truth_changed_tiles, evaluate_invalidation_coverage

device = torch.device("cuda:0")

# 1. Create Baseline Scene A (5,000 Gaussians)
scene_a = generate_procedural_scene(5000, device=device, seed=42)

# 2. Create Updated Scene B:
# - Take Scene A
# - Modify 50 Gaussians (cluster)
# - Add 200 new Gaussians
# - Remove 100 Gaussians
# - Permute the order of Gaussians so indices do NOT match!
means_a = scene_a["means"]
scales_a = scene_a["scales"]
quats_a = scene_a["quats"]
opacities_a = scene_a["opacities"]
colors_a = scene_a["colors"]

# Cluster modification: modify Gaussians around seed point
dist_from_seed = torch.norm(means_a - means_a[100:101], dim=-1)
mod_idx = torch.topk(dist_from_seed, 50, largest=False).indices

means_b = means_a.clone()
colors_b = colors_a.clone()
means_b[mod_idx] += torch.tensor([0.15, -0.10, 0.05], device=device)
colors_b[mod_idx] = torch.tensor([1.0, 0.2, 0.2], device=device)  # turn bright red

# Remove 100 Gaussians (indices 500:600)
keep_mask = torch.ones(5000, dtype=torch.bool, device=device)
keep_mask[500:600] = False
means_b = means_b[keep_mask]
scales_b = scales_a[keep_mask]
quats_b = quats_a[keep_mask]
opacities_b = opacities_a[keep_mask]
colors_b = colors_b[keep_mask]

# Add 200 new Gaussians (small gold cluster)
new_means = torch.randn((200, 3), device=device) * 0.1 + torch.tensor([0.2, 0.2, 2.3], device=device)
new_scales = torch.ones((200, 3), device=device) * 0.015
new_quats = torch.zeros((200, 4), device=device)
new_quats[:, 0] = 1.0
new_opacities = torch.ones(200, device=device) * 0.9
new_colors = torch.tensor([[0.9, 0.8, 0.1]], device=device).repeat(200, 1)

means_b = torch.cat([means_b, new_means], dim=0)
scales_b = torch.cat([scales_b, new_scales], dim=0)
quats_b = torch.cat([quats_b, new_quats], dim=0)
opacities_b = torch.cat([opacities_b, new_opacities], dim=0)
colors_b = torch.cat([colors_b, new_colors], dim=0)

# Randomly permute Scene B so row indices completely differ
perm = torch.randperm(means_b.shape[0], device=device)
means_b = means_b[perm]
scales_b = scales_b[perm]
quats_b = quats_b[perm]
opacities_b = opacities_b[perm]
colors_b = colors_b[perm]

scene_b = {
    "means": means_b,
    "scales": scales_b,
    "quats": quats_b,
    "opacities": opacities_b,
    "colors": colors_b,
}

print(f"Scene A count: {scene_a['means'].shape[0]}")
print(f"Scene B count: {scene_b['means'].shape[0]}")

# Camera setup
viewmats = torch.eye(4, device=device)[None, :, :]
Ks = torch.tensor([[512.0, 0, 256.0], [0, 512.0, 256.0], [0, 0, 1.0]], device=device)[None, :, :]
width, height = 512, 512
tile_size = 16
tile_w = width // tile_size
tile_h = height // tile_size

# Render Scene A & Scene B Ground Truths
img_a, _, _ = gsplat.rasterization(
    scene_a["means"], scene_a["quats"], scene_a["scales"],
    scene_a["opacities"], scene_a["colors"],
    viewmats, Ks, width, height
)
img_b_gt, _, _ = gsplat.rasterization(
    scene_b["means"], scene_b["quats"], scene_b["scales"],
    scene_b["opacities"], scene_b["colors"],
    viewmats, Ks, width, height
)
img_a = img_a[0]
img_b_gt = img_b_gt[0]

# --- 3D GPU Spatial Correspondence Matching ---
t0_match = time.perf_counter()
pos_tol = 0.04
feat_tol = 0.10

# Forward match: for each i in A, find nearest j in B
pos_a = scene_a["means"]
pos_b = scene_b["means"]

# Chunked nearest neighbor search to scale to 100k
chunk_size = 2000
nearest_b_for_a = torch.empty(pos_a.shape[0], dtype=torch.long, device=device)
dist_b_for_a = torch.empty(pos_a.shape[0], dtype=torch.float32, device=device)

for i in range(0, pos_a.shape[0], chunk_size):
    chunk = pos_a[i:i + chunk_size]
    dists = torch.cdist(chunk, pos_b)
    m_dist, m_idx = torch.min(dists, dim=-1)
    nearest_b_for_a[i:i + chunk_size] = m_idx
    dist_b_for_a[i:i + chunk_size] = m_dist

# Reverse match: for each j in B, find nearest i in A
nearest_a_for_b = torch.empty(pos_b.shape[0], dtype=torch.long, device=device)
dist_a_for_b = torch.empty(pos_b.shape[0], dtype=torch.float32, device=device)

for j in range(0, pos_b.shape[0], chunk_size):
    chunk = pos_b[j:j + chunk_size]
    dists = torch.cdist(chunk, pos_a)
    m_dist, m_idx = torch.min(dists, dim=-1)
    nearest_a_for_b[j:j + chunk_size] = m_idx
    dist_a_for_b[j:j + chunk_size] = m_dist

# Classify A:
# If dist_b_for_a > pos_tol => Removed in B
# If dist_b_for_a <= pos_tol => Check attribute diff
matched_b_idx = nearest_b_for_a
pos_matched_mask_a = (dist_b_for_a <= pos_tol)

# Color delta
col_delta = torch.norm(scene_a["colors"] - scene_b["colors"][matched_b_idx], dim=-1)
opac_delta = (scene_a["opacities"] - scene_b["opacities"][matched_b_idx]).abs()
scale_delta = torch.norm(scene_a["scales"] - scene_b["scales"][matched_b_idx], dim=-1)

feat_matched_mask = (col_delta <= feat_tol) & (opac_delta <= feat_tol) & (scale_delta <= feat_tol)

static_a_mask = pos_matched_mask_a & feat_matched_mask
modified_a_mask = pos_matched_mask_a & (~feat_matched_mask)
removed_a_mask = ~pos_matched_mask_a

# Classify B:
# If dist_a_for_b > pos_tol => Added in B
added_b_mask = (dist_a_for_b > pos_tol)
# For points in B matching modified points in A:
modified_b_mask = torch.zeros(pos_b.shape[0], dtype=torch.bool, device=device)
modified_b_mask[matched_b_idx[modified_a_mask]] = True

t_match = (time.perf_counter() - t0_match) * 1000.0

n_static = int(static_a_mask.sum().item())
n_mod = int(modified_a_mask.sum().item())
n_rem = int(removed_a_mask.sum().item())
n_add = int(added_b_mask.sum().item())

print(f"Matching Time: {t_match:.2f} ms")
print(f"Classification: Static={n_static}, Modified={n_mod}, Removed={n_rem}, Added={n_add}")

# --- Screen-Space Invalidation Mask ---
# Active in A that must be invalidated: removed_a + modified_a
dirty_a_idx = torch.where(removed_a_mask | modified_a_mask)[0]
# Active in B that must be invalidated: added_b + modified_b
dirty_b_idx = torch.where(added_b_mask | modified_b_mask)[0]

print(f"Dirty indices count: A={dirty_a_idx.numel()}, B={dirty_b_idx.numel()}")

# Project dirty A
rad_a, m2d_a, dep_a, _, _ = gsplat.fully_fused_projection(
    scene_a["means"][dirty_a_idx], None, scene_a["quats"][dirty_a_idx], scene_a["scales"][dirty_a_idx],
    viewmats, Ks, width, height, packed=False
)
# Project dirty B
rad_b, m2d_b, dep_b, _, _ = gsplat.fully_fused_projection(
    scene_b["means"][dirty_b_idx], None, scene_b["quats"][dirty_b_idx], scene_b["scales"][dirty_b_idx],
    viewmats, Ks, width, height, packed=False
)

from src.scene_cache import _fused_tile_invalidation_kernel
y_grid = torch.arange(tile_h, device=device).view(tile_h, 1)
x_grid = torch.arange(tile_w, device=device).view(1, tile_w)

m2d_all = torch.cat([m2d_a[0], m2d_b[0]], dim=0)
rad_all = torch.cat([rad_a[0], rad_b[0]], dim=0)
dep_all = torch.cat([dep_a[0], dep_b[0]], dim=0)

inv_mask = _fused_tile_invalidation_kernel(
    m2d_all, rad_all, dep_all, y_grid, x_grid, width, height, tile_size, tile_w, tile_h
)

# Verify against Ground Truth changed tiles
gt_tiles = compute_ground_truth_changed_tiles(img_a, img_b_gt, tile_size=tile_size, threshold=1e-3)
cov = evaluate_invalidation_coverage(inv_mask, gt_tiles)

print(f"GT Changed Tiles: {gt_tiles.sum().item()}, Predicted: {inv_mask.sum().item()}, Missed: {cov['false_negatives']}")
print(f"Zero False Negatives: {cov['zero_false_negatives']}, Recall: {cov['recall']:.4f}")

# --- Incremental Render of Scene B ---
# 1. Project all Scene B
rad_b_all, m2d_b_all, dep_b_all, con_b_all, _ = gsplat.fully_fused_projection(
    scene_b["means"], None, scene_b["quats"], scene_b["scales"],
    viewmats, Ks, width, height, packed=False
)
# 2. Intersect & Sort Scene B
_, isect_b, flat_b = gsplat.isect_tiles(m2d_b_all, rad_b_all, dep_b_all, tile_size, tile_w, tile_h)
offs_b = gsplat.isect_offset_encode(isect_b, 1, tile_w, tile_h)

# 3. Masked rasterize Scene B on dirty tiles
cols_b_sparse, _ = gsplat.rasterize_to_pixels(
    m2d_b_all, con_b_all, scene_b["colors"][None, ...], scene_b["opacities"][None, ...],
    width, height, tile_size, offs_b, flat_b, masks=inv_mask[None, ...]
)

# 4. Composite: clean tiles from cached img_a, dirty tiles from cols_b_sparse
mask_5d = inv_mask.view(tile_h, 1, tile_w, 1, 1)
cached_5d = img_a.view(tile_h, tile_size, tile_w, tile_size, 3)
sparse_5d = cols_b_sparse[0].view(tile_h, tile_size, tile_w, tile_size, 3)
comp_img = torch.where(mask_5d, sparse_5d, cached_5d).reshape(height, width, 3)

# 5. Measure Parity
metrics = compute_image_metrics(img_b_gt, comp_img)
print(f"Incremental Parity vs Ground Truth B: Max Err={metrics['max_absolute_error']:.6f}, PSNR={metrics['psnr_db']:.2f} dB, SSIM={metrics['ssim']:.6f}")
