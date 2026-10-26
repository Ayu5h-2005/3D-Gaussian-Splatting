import sys
from pathlib import Path
ROOT = Path(r"D:\3dgs-incremental-research")
sys.path.insert(0, str(ROOT))

import torch
from src.ply_io import load_3dgs_ply

device = torch.device("cuda:0")
scene_a = load_3dgs_ply(ROOT / "results" / "ply_demo" / "baseline_scene.ply", device=device)
scene_b = load_3dgs_ply(ROOT / "results" / "ply_demo" / "updated_scene.ply", device=device)

# Find static points (Stage 1)
pos_a = scene_a["means"]
pos_b = scene_b["means"]
dists_ab = torch.cdist(pos_a, pos_b)
m_dist_b, m_idx_b = torch.min(dists_ab, dim=-1)
m_dist_a, m_idx_a = torch.min(dists_ab, dim=0)

is_mutual_a = (m_idx_a[m_idx_b] == torch.arange(len(pos_a), device=device))
pos_matched_a = is_mutual_a & (m_dist_b <= 0.03)

col_d = torch.norm(scene_a["colors"] - scene_b["colors"][m_idx_b], dim=-1)
opac_d = (scene_a["opacities"] - scene_b["opacities"][m_idx_b]).abs()
scale_d = torch.norm(scene_a["scales"] - scene_b["scales"][m_idx_b], dim=-1)
feat_matched_a = (col_d <= 0.08) & (opac_d <= 0.08) & (scale_d <= 0.08)

static_a = pos_matched_a & feat_matched_a
static_b = torch.zeros(len(pos_b), dtype=torch.bool, device=device)
static_b[m_idx_b[static_a]] = True

unmatched_a = torch.where(~static_a)[0]
unmatched_b = torch.where(~static_b)[0]

print(f"Unmatched A: {len(unmatched_a)}, Unmatched B: {len(unmatched_b)}")

# In unmatched B, how many have red color? (col ≈ [1.0, 0.2, 0.2])
red_ref = torch.tensor([1.0, 0.2, 0.2], device=device)
d_red = torch.norm(scene_b["colors"][unmatched_b] - red_ref[None, :], dim=-1)
red_in_b = torch.where(d_red < 0.05)[0]
print(f"Number of red Gaussians in unmatched B: {len(red_in_b)}")

# In unmatched B, how many are the gold cluster? (col ≈ [0.95, 0.80, 0.10])
gold_ref = torch.tensor([0.95, 0.80, 0.10], device=device)
d_gold = torch.norm(scene_b["colors"][unmatched_b] - gold_ref[None, :], dim=-1)
gold_in_b = torch.where(d_gold < 0.05)[0]
print(f"Number of gold Gaussians in unmatched B: {len(gold_in_b)}")
