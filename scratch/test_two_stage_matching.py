import sys
from pathlib import Path
ROOT = Path(r"D:\3dgs-incremental-research")
sys.path.insert(0, str(ROOT))

import torch
from src.baseline_3dgs import generate_procedural_scene
from src.ply_io import load_3dgs_ply
from src.scene_comparison import match_gaussians_mutual_knn

device = torch.device("cuda:0")
scene_a = load_3dgs_ply(ROOT / "results" / "ply_demo" / "baseline_scene.ply", device=device)
scene_b = load_3dgs_ply(ROOT / "results" / "ply_demo" / "updated_scene.ply", device=device)

# Recreate exact ground truth indices
scene_a_raw = generate_procedural_scene(num_gaussians=5000, device=device, seed=42)
dist_from_seed = torch.norm(scene_a_raw["means"] - scene_a_raw["means"][100:101], dim=-1)
mod_idx_a = torch.topk(dist_from_seed, 50, largest=False).indices

# Stage 1: find static points
res_s1 = match_gaussians_mutual_knn(scene_a, scene_b, pos_tol=0.03, feat_tol=0.08)
static_a_mask = res_s1["static_a_mask"]
static_b_mask = res_s1["static_b_mask"]

unmatched_a_idx = torch.where(~static_a_mask)[0]
unmatched_b_idx = torch.where(~static_b_mask)[0]

print(f"Stage 1 matched static: A={static_a_mask.sum().item()}, B={static_b_mask.sum().item()}")
print(f"Unmatched in A: {len(unmatched_a_idx)}, Unmatched in B: {len(unmatched_b_idx)}")

# Now compare unmatched A and unmatched B!
pos_un_a = scene_a["means"][unmatched_a_idx]
pos_un_b = scene_b["means"][unmatched_b_idx]

# Check distance matrix between unmatched A and unmatched B
dists_un = torch.cdist(pos_un_a, pos_un_b)
print(f"Dists between unmatched A and unmatched B: min={dists_un.min().item():.4f}, mean={dists_un.mean().item():.4f}")

# Check feature distances (scales, quats, opacities)
scales_un_a = scene_a["scales"][unmatched_a_idx]
scales_un_b = scene_b["scales"][unmatched_b_idx]
quats_un_a = scene_a["quats"][unmatched_a_idx]
quats_un_b = scene_b["quats"][unmatched_b_idx]
opac_un_a = scene_a["opacities"][unmatched_a_idx]
opac_un_b = scene_b["opacities"][unmatched_b_idx]

# If we search for nearest neighbor in unmatched B within a displacement radius (e.g. 0.35)
disp_tol = 0.35
m_dist_b, m_idx_b = torch.min(dists_un, dim=-1)
m_dist_a, m_idx_a = torch.min(dists_un, dim=0)

is_mutual = (m_idx_a[m_idx_b] == torch.arange(len(unmatched_a_idx), device=device)) & (m_dist_b <= disp_tol)

# Also check scale and opacity consistency for the match
cand_b = m_idx_b
scale_match = torch.norm(scales_un_a - scales_un_b[cand_b], dim=-1) < 0.05
opac_match = (opac_un_a - opac_un_b[cand_b]).abs() < 0.10

mod_matched_a = is_mutual & scale_match & opac_match
print(f"Stage 2 mutual matches within disp_tol {disp_tol} and feature match: {mod_matched_a.sum().item()}")

# Check how many of mod_idx_a are in mod_matched_a
matched_global_a_idx = unmatched_a_idx[mod_matched_a]
gt_mod_set = set(mod_idx_a.tolist()) - set(range(500, 600))
detected_set = set(matched_global_a_idx.tolist())

print(f"Ground truth modified count (excluding removed 500:600): {len(gt_mod_set)}")
print(f"Detected modified count: {len(detected_set)}")
print(f"Intersection (True Positives for modified): {len(gt_mod_set & detected_set)}")
print(f"False Positives for modified: {len(detected_set - gt_mod_set)}")
print(f"False Negatives for modified: {len(gt_mod_set - detected_set)}")
