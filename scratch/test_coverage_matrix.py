import sys
from pathlib import Path

ROOT = Path(r"D:\3dgs-incremental-research")
sys.path.insert(0, str(ROOT))

import torch
import gsplat
from src.baseline_3dgs import generate_procedural_scene
from src.scene_perturbation import perturb_scene
from src.scene_cache import StaticSceneCache
from src.tile_invalidation import compute_ground_truth_changed_tiles, evaluate_invalidation_coverage

device = torch.device("cuda:0")
width, height = 512, 512
tile_size = 16

for N in [5000, 100000]:
    scene = generate_procedural_scene(num_gaussians=N, device=device, seed=42)
    scene["viewmats"] = torch.eye(4, device=device)[None, :, :]
    scene["Ks"] = torch.tensor([[512.0, 0, 256.0], [0, 512.0, 256.0], [0, 0, 1.0]], device=device)[None, :, :]
    scene["width"] = width
    scene["height"] = height

    cache = StaticSceneCache(scene, tile_size=tile_size)

    for mode in ["single", "spatial_cluster", "random_subset"]:
        frac = 0.01 if mode != "single" else 1.0 / N
        perturbed, changed_idx, _ = perturb_scene(scene, fraction=frac, seed=42, mode=mode)
        
        # Ground truth
        img_gt, _, _ = gsplat.rasterization(
            perturbed["means"], perturbed["quats"], perturbed["scales"],
            perturbed["opacities"], perturbed["colors"],
            perturbed["viewmats"], perturbed["Ks"], width, height
        )
        
        rad_d, m2d_d, dep_d, con_d = cache.project_dynamic_subset(perturbed, changed_idx)
        inv_m, _ = cache.compute_invalidation_from_cache(changed_idx, rad_d, m2d_d, dep_d, generate_pixel_mask=False)
        
        gt_tiles = compute_ground_truth_changed_tiles(cache.cached_frame, img_gt[0], tile_size=tile_size, threshold=1e-3)
        cov = evaluate_invalidation_coverage(inv_m, gt_tiles)
        gt_count = int(gt_tiles.sum().item())
        inv_count = int(inv_m.sum().item())
        fn_count = cov["false_negatives"]
        print(f"N={N:<6} | {mode:<16} | GT dirty: {gt_count:<4} | Inv: {inv_count:<4} | Missed: {fn_count:<2} | Recall: {cov['recall']:.4f}")
