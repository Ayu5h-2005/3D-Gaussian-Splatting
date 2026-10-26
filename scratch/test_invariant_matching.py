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

unmatched_a_idx = torch.where(~static_a)[0]
unmatched_b_idx = torch.where(~static_b)[0]

# Check pairwise scale/quat/opac distance between unmatched A and unmatched B
scales_a_un = scene_a["scales"][unmatched_a_idx]
scales_b_un = scene_b["scales"][unmatched_b_idx]
quats_a_un = scene_a["quats"][unmatched_a_idx]
quats_b_un = scene_b["quats"][unmatched_b_idx]
opac_a_un = scene_a["opacities"][unmatched_a_idx]
opac_b_un = scene_b["opacities"][unmatched_b_idx]

# Distance in invariant feature space: scale + opacity + quat
# Scale norm:
d_scale = torch.cdist(scales_a_un, scales_b_un)
# Opacity abs diff:
d_opac = (opac_a_un[:, None] - opac_b_un[None, :]).abs()
# Quat distance (minimum of q and -q):
dot_q = torch.abs(torch.mm(quats_a_un, quats_b_un.t()))
d_quat = 1.0 - dot_q

feat_dist = d_scale * 10.0 + d_opac * 5.0 + d_quat * 2.0
# Also distance in 3D position (should be <= e.g. max expected motion 0.5 units)
d_pos = torch.cdist(pos_a[unmatched_a_idx], pos_b[unmatched_b_idx])

# Combined distance: if pos distance <= 0.35, combine feature distance and pos distance
valid_motion = (d_pos <= 0.35)
cost = torch.where(valid_motion, feat_dist + d_pos * 0.5, torch.tensor(1e6, device=device))

m_cost_b, m_match_b = torch.min(cost, dim=-1)
m_cost_a, m_match_a = torch.min(cost, dim=0)

is_mutual_mod = (m_match_a[m_match_b] == torch.arange(len(unmatched_a_idx), device=device)) & (m_cost_b < 100.0)

print(f"Mutual matches in unmatched pool: {is_mutual_mod.sum().item()}")
matched_a_global = unmatched_a_idx[is_mutual_mod]
matched_b_global = unmatched_b_idx[m_match_b[is_mutual_mod]]

# Check displacements of matched pairs:
deltas = pos_b[matched_b_global] - pos_a[matched_a_global]
mean_delta = deltas.mean(dim=0)
print(f"Mean detected displacement: {mean_delta.tolist()} (expected [0.15, -0.10, 0.05])")
std_delta = deltas.std(dim=0)
print(f"Std of detected displacement: {std_delta.tolist()}")
