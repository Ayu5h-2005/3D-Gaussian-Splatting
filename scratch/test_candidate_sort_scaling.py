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

device = torch.device("cuda:0")
width, height = 512, 512
tile_size = 16
tile_w = width // tile_size
tile_h = height // tile_size

print(f"{'N':<8} | {'Dirty%':<8} | {'Global isect (ms)':<18} | {'Subset isect (ms)':<18} | {'Global Rast (ms)':<18} | {'Masked Rast (ms)':<18}")
print("-" * 95)

for N in [5000, 10000, 25000, 50000, 100000]:
    scene = generate_procedural_scene(num_gaussians=N, device=device, seed=42)
    viewmats = torch.eye(4, device=device)[None, :, :]
    Ks = torch.tensor([[512.0, 0, 256.0], [0, 512.0, 256.0], [0, 0, 1.0]], device=device)[None, :, :]
    scene["viewmats"] = viewmats
    scene["Ks"] = Ks
    scene["width"] = width
    scene["height"] = height

    cache = StaticSceneCache(scene, tile_size=tile_size)
    
    # Perturb 1% spatial cluster
    perturbed, changed_idx, _ = perturb_scene(
        scene, fraction=0.01, delta=(0.15, -0.10, 0.05), seed=42, mode="spatial_cluster"
    )
    
    rad_dyn, m2d_dyn, dep_dyn, con_dyn = cache.project_dynamic_subset(perturbed, changed_idx)
    inv_mask, _ = cache.compute_invalidation_from_cache(changed_idx, rad_dyn, m2d_dyn, dep_dyn, generate_pixel_mask=False)
    cache.update_working_buffers(changed_idx, rad_dyn, m2d_dyn, dep_dyn, con_dyn)
    
    dirty_pct = (inv_mask.sum().item() / 1024.0) * 100.0

    # 1. Global isect on full working buffers (all N Gaussians)
    reps = 30
    starts_g = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends_g = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts_g[i].record()
        _, isect_g, flat_g = gsplat.isect_tiles(
            cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h
        )
        ends_g[i].record()
    torch.cuda.synchronize()
    t_isect_global = np.median([starts_g[i].elapsed_time(ends_g[i]) for i in range(reps)])
    offs_g = gsplat.isect_offset_encode(isect_g, 1, tile_w, tile_h)

    # 2. Subset candidate isect (using filter_candidate_gaussians and packed=True)
    cand_idx = cache.filter_candidate_gaussians(inv_mask)
    nnz = cand_idx.numel()
    m2d_sub = cache.buf_means2d[0, cand_idx].contiguous()
    rad_sub = cache.buf_radii[0, cand_idx].contiguous()
    dep_sub = cache.buf_depths[0, cand_idx].contiguous()
    image_ids = torch.zeros(nnz, device=device, dtype=torch.int64)
    gaussian_ids = cand_idx.contiguous()

    starts_s = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends_s = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts_s[i].record()
        _, isect_s, flat_s = gsplat.isect_tiles(
            m2d_sub, rad_sub, dep_sub, tile_size, tile_w, tile_h,
            sort=True, packed=True, n_images=1, image_ids=image_ids, gaussian_ids=gaussian_ids
        )
        ends_s[i].record()
    torch.cuda.synchronize()
    t_isect_subset = np.median([starts_s[i].elapsed_time(ends_s[i]) for i in range(reps)])
    offs_s = gsplat.isect_offset_encode(isect_s, 1, tile_w, tile_h)

    # 3. Global Rasterization vs Masked Rasterization
    pert_cols = perturbed["colors"][None, ...]
    pert_opac = perturbed["opacities"][None, ...]

    starts_rg = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends_rg = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts_rg[i].record()
        _, _ = gsplat.rasterize_to_pixels(
            cache.buf_means2d, cache.buf_conics, pert_cols, pert_opac,
            width, height, tile_size, offs_g, flat_g
        )
        ends_rg[i].record()
    torch.cuda.synchronize()
    t_rast_global = np.median([starts_rg[i].elapsed_time(ends_rg[i]) for i in range(reps)])

    starts_rm = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends_rm = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts_rm[i].record()
        _, _ = gsplat.rasterize_to_pixels(
            cache.buf_means2d, cache.buf_conics, pert_cols, pert_opac,
            width, height, tile_size, offs_g, flat_g, masks=inv_mask[None, ...]
        )
        ends_rm[i].record()
    torch.cuda.synchronize()
    t_rast_masked = np.median([starts_rm[i].elapsed_time(ends_rm[i]) for i in range(reps)])

    print(f"{N:<8} | {dirty_pct:<6.1f}% | {t_isect_global:<18.3f} | {t_isect_subset:<18.3f} | {t_rast_global:<18.3f} | {t_rast_masked:<18.3f}")
