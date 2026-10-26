"""
generate_sample_ply_pair.py - Generates sample baseline & updated 3DGS PLY files for demo & testing.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from src.baseline_3dgs import generate_procedural_scene
from src.ply_io import save_3dgs_ply

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
out_dir = ROOT / "results" / "ply_demo"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Baseline Scene (5,000 Gaussians)
scene_a = generate_procedural_scene(num_gaussians=5000, device=device, seed=42)
path_a = out_dir / "baseline_scene.ply"
save_3dgs_ply(scene_a, path_a, as_inria_format=True)
print(f"Saved baseline PLY ({scene_a['means'].shape[0]} Gaussians) to {path_a}")

# 2. Updated Scene:
means_a = scene_a["means"]
scales_a = scene_a["scales"]
quats_a = scene_a["quats"]
opacities_a = scene_a["opacities"]
colors_a = scene_a["colors"]

# Displace 50 Gaussians around seed
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

# Add 150 new Gaussians (small gold cluster)
new_means = torch.randn((150, 3), device=device) * 0.08 + torch.tensor([0.25, 0.20, 2.30], device=device)
new_scales = torch.ones((150, 3), device=device) * 0.015
new_quats = torch.zeros((150, 4), device=device)
new_quats[:, 0] = 1.0
new_opacities = torch.ones(150, device=device) * 0.90
new_colors = torch.tensor([[0.95, 0.80, 0.10]], device=device).repeat(150, 1)

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

path_b = out_dir / "updated_scene.ply"
save_3dgs_ply(scene_b, path_b, as_inria_format=True)
print(f"Saved updated PLY ({scene_b['means'].shape[0]} Gaussians) to {path_b}")
