import sys
import time
from pathlib import Path

ROOT = Path(r"D:\3dgs-incremental-research")
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import gsplat
from src.scene_perturbation import perturb_scene
from src.scene_cache import StaticSceneCache

device = torch.device("cuda:0")
scene = torch.load(ROOT / "results" / "procedural_scene_5k.pt", weights_only=False)
for k, v in scene.items():
    if isinstance(v, torch.Tensor):
        scene[k] = v.to(device)

width, height = int(scene["width"]), int(scene["height"])
tile_size = 16
tile_w = width // tile_size
tile_h = height // tile_size

cache = StaticSceneCache(scene, tile_size=tile_size)
y_grid = cache.y_grid
x_grid = cache.x_grid

perturbed, changed_idx, _ = perturb_scene(
    scene, fraction=0.01, delta=(0.15, -0.10, 0.05), seed=42, mode="spatial_cluster"
)
pert_colors = perturbed["colors"][None, ...]
pert_opacities = perturbed["opacities"][None, ...]

@torch.jit.script
def fused_invalidation_tile_only(
    cached_m2d: torch.Tensor,
    cached_rad: torch.Tensor,
    cached_dep: torch.Tensor,
    dyn_idx: torch.Tensor,
    rad_dyn: torch.Tensor,
    m2d_dyn: torch.Tensor,
    dep_dyn: torch.Tensor,
    y_grid: torch.Tensor,
    x_grid: torch.Tensor,
    width: int,
    height: int,
    tile_size: int,
    tile_w: int,
    tile_h: int,
) -> torch.Tensor:
    if dyn_idx.numel() == 0:
        return torch.zeros((tile_h, tile_w), dtype=torch.bool, device=cached_m2d.device)

    m2d_old = cached_m2d[0, dyn_idx]
    rad_old = cached_rad[0, dyn_idx]
    dep_old = cached_dep[0, dyn_idx]

    m2d_all = torch.cat([m2d_old, m2d_dyn[0]], dim=0)
    rad_all = torch.cat([rad_old, rad_dyn[0]], dim=0)
    dep_all = torch.cat([dep_old, dep_dyn[0]], dim=0)

    valid = (dep_all > 0.01) & (rad_all[:, 0] > 0) & (rad_all[:, 1] > 0)
    xy_min = m2d_all - rad_all
    xy_max = m2d_all + rad_all
    in_screen = (xy_max[:, 0] >= 0) & (xy_max[:, 1] >= 0) & (xy_min[:, 0] < width) & (xy_min[:, 1] < height)
    active = (valid & in_screen).view(1, -1).float()

    uv_min_x = torch.clamp((xy_min[:, 0] / tile_size).floor().long(), 0, tile_w - 1).view(-1, 1)
    uv_max_x = torch.clamp((xy_max[:, 0] / tile_size).floor().long(), 0, tile_w - 1).view(-1, 1)
    uv_min_y = torch.clamp((xy_min[:, 1] / tile_size).floor().long(), 0, tile_h - 1).view(1, -1)
    uv_max_y = torch.clamp((xy_max[:, 1] / tile_size).floor().long(), 0, tile_h - 1).view(1, -1)

    Y = ((y_grid >= uv_min_y) & (y_grid <= uv_max_y)).float() * active
    X = ((x_grid >= uv_min_x) & (x_grid <= uv_max_x)).float()

    return torch.mm(Y, X) > 0

@torch.jit.script
def composite_tile_broadcast(
    cached_frame: torch.Tensor,
    sparse_frame: torch.Tensor,
    invalid_tile_mask: torch.Tensor,
    tile_h: int,
    tile_w: int,
    tile_size: int,
) -> torch.Tensor:
    mask_5d = invalid_tile_mask.view(tile_h, 1, tile_w, 1, 1)
    cached_5d = cached_frame.view(tile_h, tile_size, tile_w, tile_size, 3)
    sparse_5d = sparse_frame.view(tile_h, tile_size, tile_w, tile_size, 3)
    out_5d = torch.where(mask_5d, sparse_5d, cached_5d)
    return out_5d.reshape(tile_h * tile_size, tile_w * tile_size, 3)

def run_pipeline_fast():
    rad_d, m2d_d, dep_d, con_d = cache.project_dynamic_subset(perturbed, changed_idx)
    inv_m = fused_invalidation_tile_only(
        cache.cached_means2d, cache.cached_radii, cache.cached_depths,
        changed_idx, rad_d, m2d_d, dep_d,
        y_grid, x_grid, width, height, tile_size, tile_w, tile_h
    )
    cache.update_working_buffers(changed_idx, rad_d, m2d_d, dep_d, con_d)
    _, iscts, flts = gsplat.isect_tiles(cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h)
    ofst = gsplat.isect_offset_encode(iscts, 1, tile_w, tile_h)
    cls, _ = gsplat.rasterize_to_pixels(
        cache.buf_means2d, cache.buf_conics, pert_colors, pert_opacities,
        width, height, tile_size, ofst, flts, masks=inv_m[None, ...]
    )
    res = composite_tile_broadcast(cache.cached_frame, cls[0], inv_m, tile_h, tile_w, tile_size)
    cache.reset_working_buffers_to_cache(changed_idx)
    return res

# Warmup
for _ in range(25):
    _ = run_pipeline_fast()
torch.cuda.synchronize()

reps = 50
starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
for i in range(reps):
    starts[i].record()
    _ = run_pipeline_fast()
    ends[i].record()
torch.cuda.synchronize()

times = [starts[i].elapsed_time(ends[i]) for i in range(reps)]
print(f"FAST PIPELINE (no repeat_interleave): Median = {np.median(times):.4f} ms ± {np.std(times):.4f} ms")

# Now measure each step inside
e = [torch.cuda.Event(enable_timing=True) for _ in range(9)]
step_times = []
for _ in range(reps):
    e[0].record()
    rad_d, m2d_d, dep_d, con_d = cache.project_dynamic_subset(perturbed, changed_idx)
    e[1].record()
    inv_m = fused_invalidation_tile_only(
        cache.cached_means2d, cache.cached_radii, cache.cached_depths,
        changed_idx, rad_d, m2d_d, dep_d,
        y_grid, x_grid, width, height, tile_size, tile_w, tile_h
    )
    e[2].record()
    cache.update_working_buffers(changed_idx, rad_d, m2d_d, dep_d, con_d)
    e[3].record()
    _, iscts, flts = gsplat.isect_tiles(cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h)
    e[4].record()
    ofst = gsplat.isect_offset_encode(iscts, 1, tile_w, tile_h)
    e[5].record()
    cls, _ = gsplat.rasterize_to_pixels(
        cache.buf_means2d, cache.buf_conics, pert_colors, pert_opacities,
        width, height, tile_size, ofst, flts, masks=inv_m[None, ...]
    )
    e[6].record()
    res = composite_tile_broadcast(cache.cached_frame, cls[0], inv_m, tile_h, tile_w, tile_size)
    e[7].record()
    cache.reset_working_buffers_to_cache(changed_idx)
    e[8].record()
    torch.cuda.synchronize()
    step_times.append([e[i].elapsed_time(e[i+1]) for i in range(8)])

med_s = np.median(step_times, axis=0)
labels = [
    "1. Dynamic Proj (50 Gaussians)",
    "2. Fused Invalidation (tile-only)",
    "3. Buffer Update (in-place)",
    "4. isect_tiles (sorting)",
    "5. isect_offset_encode",
    "6. Masked Rasterization",
    "7. Broadcast Compositing",
    "8. Buffer Reset",
]
print("\n--- STEP BY STEP BREAKDOWN ---")
for i in range(8):
    print(f"{labels[i]:<35}: {med_s[i]:.4f} ms")
print(f"{'SUM OF STAGES':<35}: {np.sum(med_s):.4f} ms")
