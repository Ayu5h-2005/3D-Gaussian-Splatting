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

print(f"{'N':<7} | {'Mode':<15} | {'CandCount':<10} | {'Full (ms)':<10} | {'IncrGlob':<10} | {'IncrCand':<10} | {'PSNR Glob':<10} | {'PSNR Cand':<10}")
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
    
    # Ground truth full render
    img_gt, _, _ = gsplat.rasterization(
        perturbed["means"], perturbed["quats"], perturbed["scales"],
        perturbed["opacities"], perturbed["colors"],
        perturbed["viewmats"], perturbed["Ks"], width, height
    )
    img_gt = img_gt[0]

    rad_dyn, m2d_dyn, dep_dyn, con_dyn = cache.project_dynamic_subset(perturbed, changed_idx)
    inv_mask, _ = cache.compute_invalidation_from_cache(changed_idx, rad_dyn, m2d_dyn, dep_dyn, generate_pixel_mask=False)
    cache.update_working_buffers(changed_idx, rad_dyn, m2d_dyn, dep_dyn, con_dyn)

    pert_cols = perturbed["colors"][None, ...]
    pert_opac = perturbed["opacities"][None, ...]

    # Method 1: Global sort
    _, isect_g, flat_g = gsplat.isect_tiles(
        cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h
    )
    offs_g = gsplat.isect_offset_encode(isect_g, 1, tile_w, tile_h)
    cols_g, _ = gsplat.rasterize_to_pixels(
        cache.buf_means2d, cache.buf_conics, pert_cols, pert_opac,
        width, height, tile_size, offs_g, flat_g, masks=inv_mask[None, ...]
    )
    comp_g = cache.composite(cols_g[0], inv_mask)
    met_g = compute_image_metrics(img_gt, comp_g)

    # Method 2: Candidate sort
    cand_idx = cache.filter_candidate_gaussians(inv_mask)
    nnz = cand_idx.numel()
    m2d_sub = cache.buf_means2d[0, cand_idx].contiguous()
    rad_sub = cache.buf_radii[0, cand_idx].contiguous()
    dep_sub = cache.buf_depths[0, cand_idx].contiguous()
    image_ids = torch.zeros(nnz, device=device, dtype=torch.int64)
    gaussian_ids = cand_idx.contiguous()

    _, isect_s, flat_s = gsplat.isect_tiles(
        m2d_sub, rad_sub, dep_sub, tile_size, tile_w, tile_h,
        sort=True, packed=True, n_images=1, image_ids=image_ids, gaussian_ids=gaussian_ids
    )
    offs_s = gsplat.isect_offset_encode(isect_s, 1, tile_w, tile_h)
    flat_s_mapped = cand_idx[flat_s.long()].int()
    cols_s, _ = gsplat.rasterize_to_pixels(
        cache.buf_means2d, cache.buf_conics, pert_cols, pert_opac,
        width, height, tile_size, offs_s, flat_s_mapped, masks=inv_mask[None, ...]
    )
    comp_s = cache.composite(cols_s[0], inv_mask)
    met_s = compute_image_metrics(img_gt, comp_s)

    # Timing
    reps = 30
    
    # Full render timing
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts[i].record()
        gsplat.rasterization(
            perturbed["means"], perturbed["quats"], perturbed["scales"],
            perturbed["opacities"], perturbed["colors"],
            perturbed["viewmats"], perturbed["Ks"], width, height
        )
        ends[i].record()
    torch.cuda.synchronize()
    t_full = np.median([starts[i].elapsed_time(ends[i]) for i in range(reps)])

    # Incr Global timing
    starts_ig = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends_ig = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts_ig[i].record()
        r_d, m_d, d_d, c_d = cache.project_dynamic_subset(perturbed, changed_idx)
        inv_m, _ = cache.compute_invalidation_from_cache(changed_idx, r_d, m_d, d_d, generate_pixel_mask=False)
        cache.update_working_buffers(changed_idx, r_d, m_d, d_d, c_d)
        _, isect, flat = gsplat.isect_tiles(cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h)
        offs = gsplat.isect_offset_encode(isect, 1, tile_w, tile_h)
        cols, _ = gsplat.rasterize_to_pixels(
            cache.buf_means2d, cache.buf_conics, pert_cols, pert_opac,
            width, height, tile_size, offs, flat, masks=inv_m[None, ...]
        )
        cache.composite(cols[0], inv_m)
        ends_ig[i].record()
    torch.cuda.synchronize()
    t_ig = np.median([starts_ig[i].elapsed_time(ends_ig[i]) for i in range(reps)])

    # Incr Candidate timing
    starts_ic = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends_ic = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts_ic[i].record()
        r_d, m_d, d_d, c_d = cache.project_dynamic_subset(perturbed, changed_idx)
        inv_m, _ = cache.compute_invalidation_from_cache(changed_idx, r_d, m_d, d_d, generate_pixel_mask=False)
        cache.update_working_buffers(changed_idx, r_d, m_d, d_d, c_d)
        c_idx = cache.filter_candidate_gaussians(inv_m)
        n_c = c_idx.numel()
        m_s = cache.buf_means2d[0, c_idx].contiguous()
        r_s = cache.buf_radii[0, c_idx].contiguous()
        d_s = cache.buf_depths[0, c_idx].contiguous()
        im_ids = torch.zeros(n_c, device=device, dtype=torch.int64)
        g_ids = c_idx.contiguous()
        _, isect_c, flat_c = gsplat.isect_tiles(
            m_s, r_s, d_s, tile_size, tile_w, tile_h,
            sort=True, packed=True, n_images=1, image_ids=im_ids, gaussian_ids=g_ids
        )
        offs_c = gsplat.isect_offset_encode(isect_c, 1, tile_w, tile_h)
        flat_c_mapped = c_idx[flat_c.long()].int()
        cols_c, _ = gsplat.rasterize_to_pixels(
            cache.buf_means2d, cache.buf_conics, pert_cols, pert_opac,
            width, height, tile_size, offs_c, flat_c_mapped, masks=inv_m[None, ...]
        )
        cache.composite(cols_c[0], inv_m)
        ends_ic[i].record()
    torch.cuda.synchronize()
    t_ic = np.median([starts_ic[i].elapsed_time(ends_ic[i]) for i in range(reps)])

    psnr_g_str = "inf" if np.isinf(met_g["psnr_db"]) else f"{met_g['psnr_db']:.2f}"
    psnr_s_str = "inf" if np.isinf(met_s["psnr_db"]) else f"{met_s['psnr_db']:.2f}"
    print(f"{N:<7} | {'1% cluster':<15} | {nnz:<10} | {t_full:<10.3f} | {t_ig:<10.3f} | {t_ic:<10.3f} | {psnr_g_str:<10} | {psnr_s_str:<10}")
