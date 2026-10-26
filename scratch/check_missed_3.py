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

seed_scene = generate_procedural_scene(5000, device=device, seed=42)
dist_from_seed = torch.norm(seed_scene["means"] - seed_scene["means"][100:101], dim=-1)
mod_idx_50 = set(torch.topk(dist_from_seed, 50, largest=False).indices.tolist())
rem_idx_100 = set(range(500, 600))
gt_mod_a = mod_idx_50 - rem_idx_100

match_res = match_gaussians_mutual_knn(scene_a, scene_b, pos_tol=0.03, feat_tol=0.08, disp_tol=0.35)
det_mod_a = set(torch.where(match_res["modified_a_mask"])[0].tolist())

missed = gt_mod_a - det_mod_a
print("Missed modified indices in A:", missed)
for m in missed:
    # where was m classified?
    is_rem = match_res["removed_a_mask"][m].item()
    is_stat = match_res["static_a_mask"][m].item()
    print(f"  Index {m}: is_removed={is_rem}, is_static={is_stat}")
