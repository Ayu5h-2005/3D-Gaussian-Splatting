import sys
from pathlib import Path
ROOT = Path(r"D:\3dgs-incremental-research")
sys.path.insert(0, str(ROOT))

import torch
from src.ply_io import load_3dgs_ply
from src.scene_comparison import match_gaussians_mutual_knn

device = torch.device("cuda:0")
scene_a = load_3dgs_ply(ROOT / "results" / "ply_demo" / "baseline_scene.ply", device=device)
scene_b = load_3dgs_ply(ROOT / "results" / "ply_demo" / "updated_scene.ply", device=device)

for p_tol in [0.03, 0.05, 0.10, 0.15, 0.20, 0.25]:
    res = match_gaussians_mutual_knn(scene_a, scene_b, pos_tol=p_tol, feat_tol=0.08)
    print(f"pos_tol={p_tol:.2f}: static={res['num_static']}, mod={res['num_modified']}, rem={res['num_removed']}, add={res['num_added']}")
