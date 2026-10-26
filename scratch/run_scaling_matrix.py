import sys
import time
from pathlib import Path

ROOT = Path(r"D:\3dgs-incremental-research")
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import gsplat
from src.baseline_3dgs import generate_procedural_scene
from src.scene_perturbation import perturb_scene
from src.scene_cache import StaticSceneCache
from src.tile_invalidation import compute_image_metrics

device = torch.device("cuda:0")
width, height = 512, 512
tile_size = 16
tile_w = width // tile_size
tile_h = height // tile_size

print(f"{'N':<7} | {'Config':<20} | {'Full (ms)':<10} | {'Incr (ms)':<10} | {'Speedup':<8} | {'VRAM (MB)':<10} | {'Dirty%':<8} | {'PSNR':<8}")
print("-" * 100)

for N in [5000, 10000, 25000, 50000, 100000]:
    scene = generate_procedural_scene(num_gaussians=N, device=device, seed=42)
    viewmats = torch.eye(4, device=device)[None, :, :]
    Ks = torch.tensor([[512.0, 0, 256.0], [0, 512.0, 256.0], [0, 0, 1.0]], device=device)[None, :, :]
    scene["viewmats"] = viewmats
    scene["Ks"] = Ks
    scene["width"] = width
    scene["height"] = height

    torch.cuda.reset_peak_memory_stats()
    cache = StaticSceneCache(scene, tile_size=tile_size)
    
    test_configs = [
        {"mode": "single", "frac": 1.0 / N, "name": "Single Gaussian"},
        {"mode": "spatial_cluster", "frac": 0.01, "name": "1% Cluster"},
        {"mode": "spatial_cluster", "frac": 0.05, "name": "5% Cluster"},
        {"mode": "spatial_cluster", "frac": 0.10, "name": "10% Cluster"},
        {"mode": "random_subset", "frac": 0.01, "name": "1% Random"},
        {"mode": "random_subset", "frac": 0.05, "name": "5% Random"},
        {"mode": "random_subset", "frac": 0.10, "name": "10% Random"},
    ]

    for cfg in test_configs:
        perturbed, changed_idx, _ = perturb_scene(
            scene, fraction=cfg["frac"], delta=(0.15, -0.10, 0.05), seed=42, mode=cfg["mode"]
        )
        
        # Ground truth
        img_gt, _, _ = gsplat.rasterization(
            perturbed["means"], perturbed["quats"], perturbed["scales"],
            perturbed["opacities"], perturbed["colors"],
            perturbed["viewmats"], perturbed["Ks"], width, height
        )
        
        # Incremental pipeline run
        rad_d, m2d_d, dep_d, con_d = cache.project_dynamic_subset(perturbed, changed_idx)
        inv_m, _ = cache.compute_invalidation_from_cache(changed_idx, rad_d, m2d_d, dep_d, generate_pixel_mask=False)
        cache.update_working_buffers(changed_idx, rad_d, m2d_d, dep_d, con_d)
        
        pert_cols = perturbed["colors"][None, ...]
        pert_opac = perturbed["opacities"][None, ...]
        
        _, isect, flat = gsplat.isect_tiles(cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h)
        offs = gsplat.isect_offset_encode(isect, 1, tile_w, tile_h)
        cols, _ = gsplat.rasterize_to_pixels(
            cache.buf_means2d, cache.buf_conics, pert_cols, pert_opac,
            width, height, tile_size, offs, flat, masks=inv_m[None, ...]
        )
        comp = cache.composite(cols[0], inv_m)
        cache.reset_working_buffers_to_cache(changed_idx)
        
        met = compute_image_metrics(img_gt[0], comp)
        psnr_str = "inf" if np.isinf(met["psnr_db"]) else f"{met['psnr_db']:.1f}"
        dirty_pct = (inv_m.sum().item() / 1024.0) * 100.0

        # Benchmark Full Rerender
        reps = 30
        starts_f = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ends_f = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        for i in range(reps):
            starts_f[i].record()
            gsplat.rasterization(
                perturbed["means"], perturbed["quats"], perturbed["scales"],
                perturbed["opacities"], perturbed["colors"],
                perturbed["viewmats"], perturbed["Ks"], width, height
            )
            ends_f[i].record()
        torch.cuda.synchronize()
        t_full = np.median([starts_f[i].elapsed_time(ends_f[i]) for i in range(reps)])

        # Benchmark Incremental Pipeline
        starts_i = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ends_i = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        for i in range(reps):
            starts_i[i].record()
            r_d, m_d, d_d, c_d = cache.project_dynamic_subset(perturbed, changed_idx)
            i_m, _ = cache.compute_invalidation_from_cache(changed_idx, r_d, m_d, d_d, generate_pixel_mask=False)
            cache.update_working_buffers(changed_idx, r_d, m_d, d_d, c_d)
            _, isect, flat = gsplat.isect_tiles(cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h)
            offs = gsplat.isect_offset_encode(isect, 1, tile_w, tile_h)
            cols, _ = gsplat.rasterize_to_pixels(
                cache.buf_means2d, cache.buf_conics, pert_cols, pert_opac,
                width, height, tile_size, offs, flat, masks=i_m[None, ...]
            )
            cache.composite(cols[0], i_m)
            ends_i[i].record()
        torch.cuda.synchronize()
        t_incr = np.median([starts_i[i].elapsed_time(ends_i[i]) for i in range(reps)])
        
        speedup = f"{t_full / t_incr:.2f}x"
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"{N:<7} | {cfg['name']:<20} | {t_full:<10.3f} | {t_incr:<10.3f} | {speedup:<8} | {vram_mb:<10.1f} | {dirty_pct:<8.1f} | {psnr_str:<8}")
