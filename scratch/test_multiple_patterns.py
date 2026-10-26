import sys
from pathlib import Path
ROOT = Path(r"D:\3dgs-incremental-research")
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
import gsplat
from src.baseline_3dgs import generate_procedural_scene
from src.ply_io import save_3dgs_ply, load_3dgs_ply
from src.scene_comparison import compare_and_render_two_scenes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_pattern(name, scene_a, scene_b):
    print(f"\n--- Testing Pattern: {name} ---")
    viewmats = torch.eye(4, device=device)[None, :, :]
    Ks = torch.tensor([[512.0, 0, 256.0], [0, 512.0, 256.0], [0, 0, 1.0]], device=device)[None, :, :]

    res = compare_and_render_two_scenes(
        scene_a=scene_a,
        scene_b=scene_b,
        viewmats=viewmats,
        Ks=Ks,
        width=512,
        height=512,
        tile_size=16,
        pos_tol=0.03,
        feat_tol=0.08,
        num_warmups=2,
        num_reps=5,
    )

    fid = res["fidelity"]
    tiles = res["tiles"]
    cnt = res["counts"]

    print(f"  Classification: static={cnt['matched_static']}, mod={cnt['modified']}, rem={cnt['removed']}, add={cnt['added']}")
    print(f"  Tiles: dirty={tiles['invalid_tile_count']}/{tiles['total_tiles']} ({tiles['dirty_tile_percentage']:.1f}%), GT changed={tiles['gt_changed_tiles']}, False Negatives={tiles['false_negatives']}")
    print(f"  Fidelity: Max Abs Error={fid['max_abs_error']:.8f}, MSE={fid['mse']:.8f}, PSNR={fid['psnr_db']} dB, SSIM={fid['ssim']:.6f}")

    assert tiles["false_negatives"] == 0, f"False negatives > 0 in {name}!"
    assert fid["max_abs_error"] < 1e-5, f"Max error {fid['max_abs_error']} > 1e-5 in {name}!"
    assert fid["ssim"] > 0.9999, f"SSIM {fid['ssim']} < 0.9999 in {name}!"
    print(f"  [PASS] {name} passed with ZERO FALSE NEGATIVES and PERFECT PARITY!")

# Pattern 1: Identical 5k
s1_a = generate_procedural_scene(5000, device=device, seed=42)
s1_b = generate_procedural_scene(5000, device=device, seed=42)
test_pattern("Identical 5,000 Gaussians", s1_a, s1_b)

# Pattern 2: Pure Localized Spatial Cluster Translation (50 Gaussians)
s2_a = generate_procedural_scene(5000, device=device, seed=42)
s2_b = {k: v.clone() for k, v in s2_a.items()}
d_seed = torch.norm(s2_a["means"] - s2_a["means"][100:101], dim=-1)
mod50 = torch.topk(d_seed, 50, largest=False).indices
s2_b["means"][mod50] += torch.tensor([0.15, -0.10, 0.05], device=device)
test_pattern("Localized Spatial Cluster Translation (50 g)", s2_a, s2_b)

# Pattern 3: Pure Removals (100 Gaussians removed)
s3_a = generate_procedural_scene(5000, device=device, seed=42)
keep = torch.ones(5000, dtype=torch.bool, device=device)
keep[500:600] = False
s3_b = {k: v[keep].clone() for k, v in s3_a.items()}
test_pattern("Pure Removals (100 Gaussians removed)", s3_a, s3_b)

# Pattern 4: Pure Additions (150 Gaussians added)
s4_a = generate_procedural_scene(5000, device=device, seed=42)
new_m = torch.randn((150, 3), device=device) * 0.08 + torch.tensor([0.25, 0.20, 2.30], device=device)
new_s = torch.ones((150, 3), device=device) * 0.015
new_q = torch.zeros((150, 4), device=device)
new_q[:, 0] = 1.0
new_o = torch.ones(150, device=device) * 0.90
new_c = torch.tensor([[0.95, 0.80, 0.10]], device=device).repeat(150, 1)
s4_b = {
    "means": torch.cat([s4_a["means"], new_m], dim=0),
    "scales": torch.cat([s4_a["scales"], new_s], dim=0),
    "quats": torch.cat([s4_a["quats"], new_q], dim=0),
    "opacities": torch.cat([s4_a["opacities"], new_o], dim=0),
    "colors": torch.cat([s4_a["colors"], new_c], dim=0),
}
test_pattern("Pure Additions (150 Gaussians added)", s4_a, s4_b)

# Pattern 5: 10,000 Gaussians scaling with 100 Gaussians modified
s5_a = generate_procedural_scene(10000, device=device, seed=42)
s5_b = {k: v.clone() for k, v in s5_a.items()}
d_seed10k = torch.norm(s5_a["means"] - s5_a["means"][200:201], dim=-1)
mod100 = torch.topk(d_seed10k, 100, largest=False).indices
s5_b["means"][mod100] += torch.tensor([0.10, -0.05, 0.05], device=device)
s5_b["colors"][mod100] = torch.tensor([0.2, 0.9, 0.2], device=device)
test_pattern("10,000 Gaussians (100 modified + recolored)", s5_a, s5_b)
