import sys
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
N = 100000

scene = generate_procedural_scene(num_gaussians=N, device=device, seed=42)
scene["viewmats"] = torch.eye(4, device=device)[None, :, :]
scene["Ks"] = torch.tensor([[512.0, 0, 256.0], [0, 512.0, 256.0], [0, 0, 1.0]], device=device)[None, :, :]
scene["width"] = width
scene["height"] = height

cache = StaticSceneCache(scene, tile_size=tile_size)
perturbed, changed_idx, _ = perturb_scene(scene, fraction=0.01, seed=42, mode="spatial_cluster")

pert_cols = perturbed["colors"][None, ...]
pert_opac = perturbed["opacities"][None, ...]

rad_d, m2d_d, dep_d, con_d = cache.project_dynamic_subset(perturbed, changed_idx)
inv_m, _ = cache.compute_invalidation_from_cache(changed_idx, rad_d, m2d_d, dep_d, generate_pixel_mask=False)
cache.update_working_buffers(changed_idx, rad_d, m2d_d, dep_d, con_d)

reps = 40

def bench(fn):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return np.median([starts[i].elapsed_time(ends[i]) for i in range(reps)])

print(f"--- Timing breakdown for N = {N} ---")
print("1. dyn proj:", bench(lambda: cache.project_dynamic_subset(perturbed, changed_idx)))
print("2. inv mask:", bench(lambda: cache.compute_invalidation_from_cache(changed_idx, rad_d, m2d_d, dep_d, generate_pixel_mask=False)))
print("3. buf update:", bench(lambda: cache.update_working_buffers(changed_idx, rad_d, m2d_d, dep_d, con_d)))

# Global path
print("4a. Global isect_tiles:", bench(lambda: gsplat.isect_tiles(cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h)))
_, isect_g, flat_g = gsplat.isect_tiles(cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h)
print("4b. Global isect_offset_encode:", bench(lambda: gsplat.isect_offset_encode(isect_g, 1, tile_w, tile_h)))
offs_g = gsplat.isect_offset_encode(isect_g, 1, tile_w, tile_h)
print("4c. Global masked rasterize:", bench(lambda: gsplat.rasterize_to_pixels(cache.buf_means2d, cache.buf_conics, pert_cols, pert_opac, width, height, tile_size, offs_g, flat_g, masks=inv_m[None, ...])))
cols_g, _ = gsplat.rasterize_to_pixels(cache.buf_means2d, cache.buf_conics, pert_cols, pert_opac, width, height, tile_size, offs_g, flat_g, masks=inv_m[None, ...])
print("4d. Global composite:", bench(lambda: cache.composite(cols_g[0], inv_m)))

# Candidate path
print("5a. filter_candidate_gaussians:", bench(lambda: cache.filter_candidate_gaussians(inv_m)))
c_idx = cache.filter_candidate_gaussians(inv_m)
n_c = c_idx.numel()
print(f"Candidate count: {n_c} / {N}")

print("5b. Slice buffers for candidates:", bench(lambda: (
    cache.buf_means2d[0, c_idx].contiguous(),
    cache.buf_radii[0, c_idx].contiguous(),
    cache.buf_depths[0, c_idx].contiguous()
)))
m_s = cache.buf_means2d[0, c_idx].contiguous()
r_s = cache.buf_radii[0, c_idx].contiguous()
d_s = cache.buf_depths[0, c_idx].contiguous()
im_ids = torch.zeros(n_c, device=device, dtype=torch.int64)

print("5c. Candidate isect_tiles:", bench(lambda: gsplat.isect_tiles(m_s, r_s, d_s, tile_size, tile_w, tile_h, sort=True, packed=True, n_images=1, image_ids=im_ids, gaussian_ids=c_idx)))
_, isect_c, flat_c = gsplat.isect_tiles(m_s, r_s, d_s, tile_size, tile_w, tile_h, sort=True, packed=True, n_images=1, image_ids=im_ids, gaussian_ids=c_idx)
print("5d. Candidate isect_offset_encode:", bench(lambda: gsplat.isect_offset_encode(isect_c, 1, tile_w, tile_h)))
offs_c = gsplat.isect_offset_encode(isect_c, 1, tile_w, tile_h)
print("5e. Map flat_c to global:", bench(lambda: c_idx[flat_c.long()].int()))
flat_c_mapped = c_idx[flat_c.long()].int()
print("5f. Candidate masked rasterize:", bench(lambda: gsplat.rasterize_to_pixels(cache.buf_means2d, cache.buf_conics, pert_cols, pert_opac, width, height, tile_size, offs_c, flat_c_mapped, masks=inv_m[None, ...])))
