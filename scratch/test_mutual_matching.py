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
from src.scene_cache import _fused_tile_invalidation_kernel

device = torch.device("cuda:0")

# 1. Create Baseline Scene A (5,000 Gaussians)
scene_a = generate_procedural_scene(5000, device=device, seed=42)

# 2. Create Updated Scene B:
means_a = scene_a["means"]
scales_a = scene_a["scales"]
quats_a = scene_a["quats"]
opacities_a = scene_a["opacities"]
colors_a = scene_a["colors"]

dist_from_seed = torch.norm(means_a - means_a[100:101], dim=-1)
mod_idx = torch.topk(dist_from_seed, 50, largest=False).indices

means_b = means_a.clone()
colors_b = colors_a.clone()
means_b[mod_idx] += torch.tensor([0.15, -0.10, 0.05], device=device)
colors_b[mod_idx] = torch.tensor([1.0, 0.2, 0.2], device=device)

# Remove 100 Gaussians
keep_mask = torch.ones(5000, dtype=torch.bool, device=device)
keep_mask[500:600] = False
means_b = means_b[keep_mask]
scales_b = scales_a[keep_mask]
quats_b = quats_a[keep_mask]
opacities_b = opacities_a[keep_mask]
colors_b = colors_b[keep_mask]

# Add 200 new Gaussians
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

# Randomly permute
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

# Camera setup
viewmats = torch.eye(4, device=device)[None, :, :]
Ks = torch.tensor([[512.0, 0, 256.0], [0, 512.0, 256.0], [0, 0, 1.0]], device=device)[None, :, :]
width, height = 512, 512
tile_size = 16
tile_w = width // tile_size
tile_h = height // tile_size

img_a, _, _ = gsplat.rasterization(scene_a["means"], scene_a["quats"], scene_a["scales"], scene_a["opacities"], scene_a["colors"], viewmats, Ks, width, height)
img_b_gt, _, _ = gsplat.rasterization(scene_b["means"], scene_b["quats"], scene_b["scales"], scene_b["opacities"], scene_b["colors"], viewmats, Ks, width, height)
img_a = img_a[0]
img_b_gt = img_b_gt[0]

# --- 3D GPU Mutual Spatial Correspondence Matching ---
pos_tol = 0.02
feat_tol = 0.05

pos_a = scene_a["means"]
pos_b = scene_b["means"]
na = pos_a.shape[0]
nb = pos_b.shape[0]

chunk_size = 2000
nearest_b_for_a = torch.empty(na, dtype=torch.long, device=device)
dist_b_for_a = torch.empty(na, dtype=torch.float32, device=device)

for i in range(0, na, chunk_size):
    chunk = pos_a[i:i + chunk_size]
    dists = torch.cdist(chunk, pos_b)
    m_dist, m_idx = torch.min(dists, dim=-1)
    nearest_b_for_a[i:i + chunk_size] = m_idx
    dist_b_for_a[i:i + chunk_size] = m_dist

nearest_a_for_b = torch.empty(nb, dtype=torch.long, device=device)
dist_a_for_b = torch.empty(nb, dtype=torch.float32, device=device)

for j in range(0, nb, chunk_size):
    chunk = pos_b[j:j + chunk_size]
    dists = torch.cdist(chunk, pos_a)
    m_dist, m_idx = torch.min(dists, dim=-1)
    nearest_a_for_b[j:j + chunk_size] = m_idx
    dist_a_for_b[j:j + chunk_size] = m_dist

# Mutual Nearest Neighbors
# i in A matches j in B iff nearest_b_for_a[i] == j AND nearest_a_for_b[j] == i
is_mutual_a = (nearest_a_for_b[nearest_b_for_a] == torch.arange(na, device=device))
is_mutual_b = (nearest_b_for_a[nearest_a_for_b] == torch.arange(nb, device=device))

matched_b_idx = nearest_b_for_a
pos_matched_a = is_mutual_a & (dist_b_for_a <= pos_tol)

# Feature check on mutual spatial matches
col_delta = torch.norm(scene_a["colors"] - scene_b["colors"][matched_b_idx], dim=-1)
opac_delta = (scene_a["opacities"] - scene_b["opacities"][matched_b_idx]).abs()
scale_delta = torch.norm(scene_a["scales"] - scene_b["scales"][matched_b_idx], dim=-1)
feat_matched = (col_delta <= feat_tol) & (opac_delta <= feat_tol) & (scale_delta <= feat_tol)

static_a_mask = pos_matched_a & feat_matched
# Any Gaussian in A that is NOT strictly static must be considered dirty!
dirty_a_mask = ~static_a_mask

# Similarly, in B, any Gaussian that is NOT strictly static must be considered dirty!
matched_a_idx = nearest_a_for_b
pos_matched_b = is_mutual_b & (dist_a_for_b <= pos_tol)
col_delta_b = torch.norm(scene_b["colors"] - scene_a["colors"][matched_a_idx], dim=-1)
opac_delta_b = (scene_b["opacities"] - scene_a["opacities"][matched_a_idx]).abs()
scale_delta_b = torch.norm(scene_b["scales"] - scene_a["scales"][matched_a_idx], dim=-1)
feat_matched_b = (col_delta_b <= feat_tol) & (opac_delta_b <= feat_tol) & (scale_delta_b <= feat_tol)

static_b_mask = pos_matched_b & feat_matched_b
dirty_b_mask = ~static_b_mask

print(f"Scene A: Total={na}, Static={static_a_mask.sum().item()}, Dirty={dirty_a_mask.sum().item()}")
print(f"Scene B: Total={nb}, Static={static_b_mask.sum().item()}, Dirty={dirty_b_mask.sum().item()}")

dirty_a_idx = torch.where(dirty_a_mask)[0]
dirty_b_idx = torch.where(dirty_b_mask)[0]

# Project dirty A & B
rad_a, m2d_a, dep_a, _, _ = gsplat.fully_fused_projection(
    scene_a["means"][dirty_a_idx], None, scene_a["quats"][dirty_a_idx], scene_a["scales"][dirty_a_idx],
    viewmats, Ks, width, height, packed=False
)
rad_b, m2d_b, dep_b, _, _ = gsplat.fully_fused_projection(
    scene_b["means"][dirty_b_idx], None, scene_b["quats"][dirty_b_idx], scene_b["scales"][dirty_b_idx],
    viewmats, Ks, width, height, packed=False
)

y_grid = torch.arange(tile_h, device=device).view(tile_h, 1)
x_grid = torch.arange(tile_w, device=device).view(1, tile_w)

m2d_all = torch.cat([m2d_a[0], m2d_b[0]], dim=0)
rad_all = torch.cat([rad_a[0], rad_b[0]], dim=0)
dep_all = torch.cat([dep_a[0], dep_b[0]], dim=0)

inv_mask = _fused_tile_invalidation_kernel(
    m2d_all, rad_all, dep_all, y_grid, x_grid, width, height, tile_size, tile_w, tile_h
)

# Ground truth comparison
gt_tiles = compute_ground_truth_changed_tiles(img_a, img_b_gt, tile_size=tile_size, threshold=1e-3)
cov = evaluate_invalidation_coverage(inv_mask, gt_tiles)

print(f"GT Changed Tiles: {gt_tiles.sum().item()}, Predicted: {inv_mask.sum().item()}, Missed: {cov['false_negatives']}")
print(f"Zero False Negatives: {cov['zero_false_negatives']}, Recall: {cov['recall']:.4f}")

# Incremental Render
rad_b_all, m2d_b_all, dep_b_all, con_b_all, _ = gsplat.fully_fused_projection(
    scene_b["means"], None, scene_b["quats"], scene_b["scales"],
    viewmats, Ks, width, height, packed=False
)
_, isect_b, flat_b = gsplat.isect_tiles(m2d_b_all, rad_b_all, dep_b_all, tile_size, tile_w, tile_h)
offs_b = gsplat.isect_offset_encode(isect_b, 1, tile_w, tile_h)

cols_b_sparse, _ = gsplat.rasterize_to_pixels(
    m2d_b_all, con_b_all, scene_b["colors"][None, ...], scene_b["opacities"][None, ...],
    width, height, tile_size, offs_b, flat_b, masks=inv_mask[None, ...]
)

mask_5d = inv_mask.view(tile_h, 1, tile_w, 1, 1)
cached_5d = img_a.view(tile_h, tile_size, tile_w, tile_size, 3)
sparse_5d = cols_b_sparse[0].view(tile_h, tile_size, tile_w, tile_size, 3)
comp_img = torch.where(mask_5d, sparse_5d, cached_5d).reshape(height, width, 3)

metrics = compute_image_metrics(img_b_gt, comp_img)
psnr_s = "inf" if np.isinf(metrics["psnr_db"]) else f"{metrics['psnr_db']:.2f}"
print(f"Incremental Parity vs Ground Truth B: Max Err={metrics['max_absolute_error']:.8f}, PSNR={psnr_s} dB, SSIM={metrics['ssim']:.6f}")
