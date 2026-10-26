import sys
from pathlib import Path

ROOT = Path(r"D:\3dgs-incremental-research")
sys.path.insert(0, str(ROOT))

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

scene = generate_procedural_scene(num_gaussians=5000, device=device, seed=42)
viewmats = torch.eye(4, device=device)[None, :, :]
Ks = torch.tensor([[512.0, 0, 256.0], [0, 512.0, 256.0], [0, 0, 1.0]], device=device)[None, :, :]
scene["viewmats"] = viewmats
scene["Ks"] = Ks
scene["width"] = width
scene["height"] = height

cache = StaticSceneCache(scene, tile_size=tile_size)

perturbed, changed_idx, _ = perturb_scene(
    scene, fraction=0.01, delta=(0.15, -0.10, 0.05), seed=42, mode="spatial_cluster"
)

rad_dyn, m2d_dyn, dep_dyn, con_dyn = cache.project_dynamic_subset(perturbed, changed_idx)
inv_mask, _ = cache.compute_invalidation_from_cache(changed_idx, rad_dyn, m2d_dyn, dep_dyn, generate_pixel_mask=False)
cache.update_working_buffers(changed_idx, rad_dyn, m2d_dyn, dep_dyn, con_dyn)

pert_cols = perturbed["colors"][None, ...]
pert_opac = perturbed["opacities"][None, ...]

# Global isect
_, isect_g, flat_g = gsplat.isect_tiles(
    cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h
)
offs_g = gsplat.isect_offset_encode(isect_g, 1, tile_w, tile_h)
cols_g, _ = gsplat.rasterize_to_pixels(
    cache.buf_means2d, cache.buf_conics, pert_cols, pert_opac,
    width, height, tile_size, offs_g, flat_g, masks=inv_mask[None, ...]
)

# Candidate isect
cand_idx = cache.filter_candidate_gaussians(inv_mask)
print(f"Total Gaussians: 5000, Candidate Gaussians: {cand_idx.numel()}")

m2d_sub = cache.buf_means2d[0, cand_idx].contiguous()
rad_sub = cache.buf_radii[0, cand_idx].contiguous()
dep_sub = cache.buf_depths[0, cand_idx].contiguous()
image_ids = torch.zeros(cand_idx.numel(), device=device, dtype=torch.int64)
gaussian_ids = cand_idx.contiguous()

_, isect_s, flat_s = gsplat.isect_tiles(
    m2d_sub, rad_sub, dep_sub, tile_size, tile_w, tile_h,
    sort=True, packed=True, n_images=1, image_ids=image_ids, gaussian_ids=gaussian_ids
)
offs_s = gsplat.isect_offset_encode(isect_s, 1, tile_w, tile_h)
cols_s, _ = gsplat.rasterize_to_pixels(
    cache.buf_means2d, cache.buf_conics, pert_cols, pert_opac,
    width, height, tile_size, offs_s, flat_s, masks=inv_mask[None, ...]
)

diff = (cols_g - cols_s).abs()
print("Max diff between global and candidate raster on masked tiles:", diff.max().item())

# Where does diff occur?
diff_mask = (diff.sum(dim=-1)[0] > 1e-4)
print("Number of pixels with diff > 1e-4:", diff_mask.sum().item())

# Let's inspect isect_ids / offs for a dirty tile!
dirty_y, dirty_x = torch.where(inv_mask)
print("Dirty tiles count:", len(dirty_y))
for k in range(min(5, len(dirty_y))):
    ty, tx = dirty_y[k].item(), dirty_x[k].item()
    tile_id = ty * tile_w + tx
    start_g, end_g = offs_g[0, ty, tx].item(), offs_g[0, ty, tx+1 if tx+1 < tile_w else 0].item() # wait!
    print(f"Tile ({ty}, {tx}) - offs_g slice: {offs_g[0, ty, tx]}, offs_s slice: {offs_s[0, ty, tx]}")
