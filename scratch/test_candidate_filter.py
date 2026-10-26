import sys
from pathlib import Path

ROOT = Path(r"D:\3dgs-incremental-research")
sys.path.insert(0, str(ROOT))

import torch
import gsplat
from src.scene_perturbation import perturb_scene
from src.tile_invalidation import compute_tile_invalidation_mask, composite_with_cache, compute_image_metrics

device = torch.device("cuda:0")
scene = torch.load(ROOT / "results" / "procedural_scene_5k.pt", weights_only=False)
for k, v in scene.items():
    if isinstance(v, torch.Tensor):
        scene[k] = v.to(device)

width, height = int(scene["width"]), int(scene["height"])
tile_size = 16
tile_w, tile_h = width // tile_size, height // tile_size

perturbed, changed_idx, _ = perturb_scene(
    scene, fraction=0.01, delta=(0.12, -0.10, 0.04), seed=42, mode="spatial_cluster"
)

# Reference full render
img_ref, _, _ = gsplat.rasterization(
    perturbed["means"], perturbed["quats"], perturbed["scales"],
    perturbed["opacities"], perturbed["colors"],
    perturbed["viewmats"], perturbed["Ks"], width, height
)

# Baseline t0 render
img_t0, _, _ = gsplat.rasterization(
    scene["means"], scene["quats"], scene["scales"],
    scene["opacities"], scene["colors"],
    scene["viewmats"], scene["Ks"], width, height
)

# Invalidation mask
invalid_mask, pixel_mask, stats = compute_tile_invalidation_mask(
    scene, perturbed, changed_idx, tile_size=tile_size, compute_stats=True
)
print(f"Dirty tiles: {stats['invalid_tile_count']} / {stats['total_tiles']}")

# Full projection of perturbed scene
radii, means2d, depths, conics, _ = gsplat.fully_fused_projection(
    perturbed["means"], None, perturbed["quats"], perturbed["scales"],
    perturbed["viewmats"], perturbed["Ks"], width, height, packed=False
)

rad = radii[0]
m2d = means2d[0]
dep = depths[0]
valid = (dep > 0.01) & (rad[:, 0] > 0) & (rad[:, 1] > 0)
u_min = torch.clamp(torch.floor((m2d[:, 0] - rad[:, 0]) / tile_size).long(), 0, tile_w - 1)
u_max = torch.clamp(torch.floor((m2d[:, 0] + rad[:, 0]) / tile_size).long(), 0, tile_w - 1)
v_min = torch.clamp(torch.floor((m2d[:, 1] - rad[:, 1]) / tile_size).long(), 0, tile_h - 1)
v_max = torch.clamp(torch.floor((m2d[:, 1] + rad[:, 1]) / tile_size).long(), 0, tile_h - 1)

# 2D Integral image of invalid_mask
inv_int = invalid_mask.int()
prefix = torch.zeros((tile_h + 1, tile_w + 1), dtype=torch.int32, device=device)
prefix[1:, 1:] = torch.cumsum(torch.cumsum(inv_int, dim=0), dim=1)

# Bounding box sum query: O(1) per Gaussian on GPU!
box_dirty_count = (
    prefix[v_max + 1, u_max + 1]
    - prefix[v_min, u_max + 1]
    - prefix[v_max + 1, u_min]
    + prefix[v_min, u_min]
)
touches_dirty = (box_dirty_count > 0) & valid
cand_idx = torch.nonzero(touches_dirty).squeeze(-1)
print(f"Total Gaussians touching dirty tiles: {cand_idx.numel()} out of {scene['means'].shape[0]} ({cand_idx.numel()/scene['means'].shape[0]*100:.1f}%)")

# Run isect_tiles and rasterize_to_pixels ONLY on cand_idx!
m2d_cand = means2d[:, cand_idx]
rad_cand = radii[:, cand_idx]
dep_cand = depths[:, cand_idx]
con_cand = conics[:, cand_idx]
col_cand = perturbed["colors"][None, cand_idx]
opac_cand = perturbed["opacities"][None, cand_idx]

_, isect_ids_cand, flat_ids_cand = gsplat.isect_tiles(
    m2d_cand, rad_cand, dep_cand, tile_size, tile_w, tile_h
)
offsets_cand = gsplat.isect_offset_encode(isect_ids_cand, 1, tile_w, tile_h)
sparse_cand, _ = gsplat.rasterize_to_pixels(
    m2d_cand, con_cand, col_cand, opac_cand,
    width, height, tile_size, offsets_cand, flat_ids_cand,
    masks=invalid_mask[None, ...]
)

comp_cand = composite_with_cache(img_t0[0], sparse_cand[0], pixel_mask)

# Check fidelity
metrics = compute_image_metrics(img_ref[0], comp_cand)
print("Fidelity with candidate filtering:")
print(f"  Max abs error: {metrics['max_absolute_error']}")
print(f"  PSNR: {metrics['psnr_db']}")
print(f"  SSIM: {metrics['ssim']}")
