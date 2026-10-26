import sys
from pathlib import Path
ROOT = Path(r"D:\3dgs-incremental-research")
sys.path.insert(0, str(ROOT))

import torch
from src.baseline_3dgs import generate_procedural_scene
from src.ply_io import load_3dgs_ply

device = torch.device("cuda:0")
scene_a_ply = load_3dgs_ply(ROOT / "results" / "ply_demo" / "baseline_scene.ply", device=device)
scene_b_ply = load_3dgs_ply(ROOT / "results" / "ply_demo" / "updated_scene.ply", device=device)

# How were points generated in generate_sample_ply_pair.py?
# In generate_sample_ply_pair.py:
# scene_a is generated with seed 42.
# mod_idx = topk 50 closest to means_a[100:101].
# means_b = means_a.clone()
# means_b[mod_idx] += [0.15, -0.10, 0.05]
# colors_b[mod_idx] = [1.0, 0.2, 0.2]
# keep_mask[500:600] = False
# means_b = means_b[keep_mask] ...
# perm = torch.randperm(...)

# Let's find for each of the 49 modified points in A, where is its corresponding point in scene_b_ply?
scene_a = generate_procedural_scene(5000, device=device, seed=42)
dist_from_seed = torch.norm(scene_a["means"] - scene_a["means"][100:101], dim=-1)
mod_idx = torch.topk(dist_from_seed, 50, largest=False).indices.tolist()
valid_mod_idx = [i for i in mod_idx if not (500 <= i < 600)]

pos_a = scene_a_ply["means"]
pos_b = scene_b_ply["means"]

print(f"Number of true modified points in A: {len(valid_mod_idx)}")

for k, idx in enumerate(valid_mod_idx[:10]):
    expected_pos_b = pos_a[idx] + torch.tensor([0.15, -0.10, 0.05], device=device)
    # Find distance from expected_pos_b to all points in pos_b
    dists_to_all_b = torch.norm(pos_b - expected_pos_b[None, :], dim=-1)
    min_d, min_b_idx = torch.min(dists_to_all_b, dim=-1)
    
    # Distance from pos_a[idx] to pos_b[min_b_idx]
    actual_disp = torch.norm(pos_b[min_b_idx] - pos_a[idx]).item()
    
    # What was the closest point in pos_b to pos_a[idx]?
    dists_from_old_a = torch.norm(pos_b - pos_a[idx][None, :], dim=-1)
    closest_to_old_d, closest_to_old_b_idx = torch.min(dists_from_old_a, dim=-1)
    
    print(f"Mod Point A[{idx}]:")
    print(f"  True match in B is B[{min_b_idx}] with delta_from_expected={min_d.item():.6f}")
    print(f"  Distance from A[{idx}] to true match B[{min_b_idx}]: {actual_disp:.4f}")
    print(f"  Distance from A[{idx}] to closest point in B: {closest_to_old_d.item():.4f} (at B[{closest_to_old_b_idx}])")
    print(f"  Is closest in B the true match? {min_b_idx.item() == closest_to_old_b_idx.item()}")
