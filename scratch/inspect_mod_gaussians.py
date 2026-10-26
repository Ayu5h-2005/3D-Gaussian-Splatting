import sys
from pathlib import Path
ROOT = Path(r"D:\3dgs-incremental-research")
sys.path.insert(0, str(ROOT))

import torch
from src.ply_io import load_3dgs_ply
from src.scene_comparison import match_gaussians_mutual_knn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scene_a = load_3dgs_ply(ROOT / "results" / "ply_demo" / "baseline_scene.ply", device=device)
scene_b = load_3dgs_ply(ROOT / "results" / "ply_demo" / "updated_scene.ply", device=device)

res = match_gaussians_mutual_knn(scene_a, scene_b, pos_tol=0.03, feat_tol=0.08)

mod_a_idx = torch.where(res["modified_a_mask"])[0]
print("Indices in A classified as modified:", mod_a_idx.tolist())

pos_a = scene_a["means"]
pos_b = scene_b["means"]

for idx in mod_a_idx:
    # find matched in B
    # From match_gaussians_mutual_knn:
    chunk = pos_a[idx:idx+1]
    dists = torch.cdist(chunk, pos_b)[0]
    m_dist, m_idx = torch.min(dists, dim=-1)
    print(f"Gaussian A[{idx}]: pos={pos_a[idx].tolist()}, col={scene_a['colors'][idx].tolist()}")
    print(f"  Matched B[{m_idx}]: dist={m_dist.item():.4f}, pos={pos_b[m_idx].tolist()}, col={scene_b['colors'][m_idx].tolist()}")
    col_d = torch.norm(scene_a["colors"][idx] - scene_b["colors"][m_idx]).item()
    scale_d = torch.norm(scene_a["scales"][idx] - scene_b["scales"][m_idx]).item()
    print(f"  col_delta={col_d:.4f}, scale_delta={scale_d:.4f}")
