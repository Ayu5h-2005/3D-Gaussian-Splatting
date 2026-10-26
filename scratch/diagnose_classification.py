import sys
from pathlib import Path
ROOT = Path(r"D:\3dgs-incremental-research")
sys.path.insert(0, str(ROOT))

import torch
from src.baseline_3dgs import generate_procedural_scene
from src.ply_io import load_3dgs_ply
from src.scene_comparison import match_gaussians_mutual_knn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Recreate exact ground truth mapping from generate_sample_ply_pair.py
scene_a = generate_procedural_scene(num_gaussians=5000, device=device, seed=42)
means_a = scene_a["means"]

dist_from_seed = torch.norm(means_a - means_a[100:101], dim=-1)
mod_idx = torch.topk(dist_from_seed, 50, largest=False).indices

print(f"Mod indices count: {len(mod_idx)}, min: {mod_idx.min().item()}, max: {mod_idx.max().item()}")
overlap_with_removed = [i for i in mod_idx.tolist() if 500 <= i < 600]
print(f"Overlap between modified and removed (500:600): {len(overlap_with_removed)}")

# Load the actual saved PLYs
scene_a_loaded = load_3dgs_ply(ROOT / "results" / "ply_demo" / "baseline_scene.ply", device=device)
scene_b_loaded = load_3dgs_ply(ROOT / "results" / "ply_demo" / "updated_scene.ply", device=device)

# Run current matching
res = match_gaussians_mutual_knn(scene_a_loaded, scene_b_loaded, pos_tol=0.03, feat_tol=0.08)

print("\nCurrent matching results:")
print(f"Num static: {res['num_static']}")
print(f"Num modified: {res['num_modified']}")
print(f"Num removed: {res['num_removed']}")
print(f"Num added: {res['num_added']}")

# Check classification of ground-truth modified Gaussians in A
gt_mod_mask = torch.zeros(5000, dtype=torch.bool, device=device)
gt_mod_mask[mod_idx] = True

gt_rem_mask = torch.zeros(5000, dtype=torch.bool, device=device)
gt_rem_mask[500:600] = True

gt_static_mask = ~(gt_mod_mask | gt_rem_mask)

print("\n--- Breakdown of Ground-Truth Categories in A ---")
print(f"GT Static count: {gt_static_mask.sum().item()} (expected 4850)")
print(f"GT Modified count: {gt_mod_mask.sum().item()} (expected 50)")
print(f"GT Removed count: {gt_rem_mask.sum().item()} (expected 100)")

classified_static_a = res["static_a_mask"]
classified_mod_a = res["modified_a_mask"]
classified_rem_a = res["removed_a_mask"]

print("\nWhere did GT Static go?")
print(f"  Classified as Static: {(gt_static_mask & classified_static_a).sum().item()}")
print(f"  Classified as Modified: {(gt_static_mask & classified_mod_a).sum().item()}")
print(f"  Classified as Removed: {(gt_static_mask & classified_rem_a).sum().item()}")

print("\nWhere did GT Modified go?")
print(f"  Classified as Static: {(gt_mod_mask & classified_static_a).sum().item()}")
print(f"  Classified as Modified: {(gt_mod_mask & classified_mod_a).sum().item()}")
print(f"  Classified as Removed: {(gt_mod_mask & classified_rem_a).sum().item()}")

print("\nWhere did GT Removed go?")
print(f"  Classified as Static: {(gt_rem_mask & classified_static_a).sum().item()}")
print(f"  Classified as Modified: {(gt_rem_mask & classified_mod_a).sum().item()}")
print(f"  Classified as Removed: {(gt_rem_mask & classified_rem_a).sum().item()}")
