import sys
import time
from pathlib import Path
from typing import Tuple

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

@torch.jit.script
def fused_invalidation_fast(
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    if dyn_idx.numel() == 0:
        inv_mask = torch.zeros((tile_h, tile_w), dtype=torch.bool, device=cached_m2d.device)
        px_mask = torch.zeros((height, width, 1), dtype=torch.bool, device=cached_m2d.device)
        return inv_mask, px_mask

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

    invalid_tile_mask = torch.mm(Y, X) > 0
    invalid_pixel_mask = invalid_tile_mask.repeat_interleave(tile_size, dim=0).repeat_interleave(tile_size, dim=1)[:height, :width, None]
    return invalid_tile_mask, invalid_pixel_mask

# Warmup JIT
dummy_idx = torch.tensor([0, 1], dtype=torch.long, device=device)
r_d, m_d, d_d, _ = cache.project_dynamic_subset(scene, dummy_idx)
for _ in range(30):
    _ = fused_invalidation_fast(
        cache.cached_means2d, cache.cached_radii, cache.cached_depths,
        dummy_idx, r_d, m_d, d_d,
        y_grid, x_grid, width, height, tile_size, tile_w, tile_h
    )
torch.cuda.synchronize()

reps = 30
warmups = 15

print(f"{'Fraction':<10} | {'Mode':<16} | {'Full (ms)':<10} | {'Incr (ms)':<10} | {'Speedup':<8} | {'Inv (ms)':<10} | {'Proj (ms)':<10} | {'Dirty%':<8}")
print("-" * 95)

for frac, mode in [
    (0.0002, "single"),
    (0.01, "spatial_cluster"),
    (0.02, "spatial_cluster"),
    (0.05, "spatial_cluster"),
    (0.10, "spatial_cluster"),
    (0.25, "spatial_cluster"),
    (0.01, "random_subset"),
    (0.02, "random_subset"),
    (0.05, "random_subset"),
    (0.10, "random_subset"),
    (0.25, "random_subset"),
]:
    perturbed, changed_idx, _ = perturb_scene(
        scene, fraction=frac, delta=(0.15, -0.10, 0.05), seed=42, mode=mode
    )
    pert_colors = perturbed["colors"][None, ...]
    pert_opacities = perturbed["opacities"][None, ...]

    def run_full():
        return gsplat.rasterization(
            perturbed["means"], perturbed["quats"], perturbed["scales"],
            perturbed["opacities"], perturbed["colors"],
            perturbed["viewmats"], perturbed["Ks"], width, height
        )

    def run_incr_optimized():
        rad_d, m2d_d, dep_d, con_d = cache.project_dynamic_subset(perturbed, changed_idx)
        inv_m, px_m = fused_invalidation_fast(
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
        res = torch.where(px_m, cls[0], cache.cached_frame)
        cache.reset_working_buffers_to_cache(changed_idx)
        return res

    # Measure
    for _ in range(warmups):
        _ = run_full()
        _ = run_incr_optimized()
    torch.cuda.synchronize()

    starts_f = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends_f = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts_f[i].record()
        _ = run_full()
        ends_f[i].record()
    torch.cuda.synchronize()
    t_full = np.median([starts_f[i].elapsed_time(ends_f[i]) for i in range(reps)])

    starts_i = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends_i = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts_i[i].record()
        _ = run_incr_optimized()
        ends_i[i].record()
    torch.cuda.synchronize()
    t_incr = np.median([starts_i[i].elapsed_time(ends_i[i]) for i in range(reps)])

    # Separate measurements for inv and proj
    rad_d, m2d_d, dep_d, con_d = cache.project_dynamic_subset(perturbed, changed_idx)
    starts_inv = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends_inv = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts_inv[i].record()
        inv_m, px_m = fused_invalidation_fast(
            cache.cached_means2d, cache.cached_radii, cache.cached_depths,
            changed_idx, rad_d, m2d_d, dep_d,
            y_grid, x_grid, width, height, tile_size, tile_w, tile_h
        )
        ends_inv[i].record()
    torch.cuda.synchronize()
    t_inv = np.median([starts_inv[i].elapsed_time(ends_inv[i]) for i in range(reps)])

    starts_proj = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends_proj = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts_proj[i].record()
        _ = cache.project_dynamic_subset(perturbed, changed_idx)
        ends_proj[i].record()
    torch.cuda.synchronize()
    t_proj = np.median([starts_proj[i].elapsed_time(ends_proj[i]) for i in range(reps)])

    dirty_pct = (inv_m.sum().item() / 1024.0) * 100.0
    speedup = t_full / t_incr
    print(f"{frac*100:<9.2f}% | {mode:<16} | {t_full:<10.3f} | {t_incr:<10.3f} | {speedup:<8.2f} | {t_inv:<10.3f} | {t_proj:<10.3f} | {dirty_pct:<8.1f}%")
