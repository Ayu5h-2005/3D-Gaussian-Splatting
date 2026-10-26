import sys
from pathlib import Path
ROOT = Path(r"D:\3dgs-incremental-research")
sys.path.insert(0, str(ROOT))

import torch
from src.baseline_3dgs import generate_procedural_scene
from src.ply_io import load_3dgs_ply

device = torch.device("cuda:0")
scene_a = load_3dgs_ply(ROOT / "results" / "ply_demo" / "baseline_scene.ply", device=device)
scene_b = load_3dgs_ply(ROOT / "results" / "ply_demo" / "updated_scene.ply", device=device)

pos_a = scene_a["means"]
pos_b = scene_b["means"]

# What is average nearest-neighbor distance within Scene A?
dists_self = torch.cdist(pos_a[:500], pos_a)
dists_self.fill_diagonal_(1e9)
min_self = torch.min(dists_self, dim=-1).values
print(f"Within Scene A: Mean NN dist: {min_self.mean().item():.4f}, Median NN dist: {min_self.median().item():.4f}, Min: {min_self.min().item():.4f}")
