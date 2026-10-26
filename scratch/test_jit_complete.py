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
perturbed, changed_idx, _ = perturb_scene(
    scene, fraction=0.01, delta=(0.15, -0.10, 0.05), seed=42, mode="spatial_cluster"
)
rad_dyn, m2d_dyn, dep_dyn, con_dyn = cache.project_dynamic_subset(perturbed, changed_idx)

y_grid = cache.y_grid
x_grid = cache.x_grid

@torch.jit.script
def fused_invalidation_complete(
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

    m2d_new = m2d_dyn[0]
    rad_new = rad_dyn[0]
    dep_new = dep_dyn[0]

    m2d_all = torch.cat([m2d_old, m2d_new], dim=0)
    rad_all = torch.cat([rad_old, rad_new], dim=0)
    dep_all = torch.cat([dep_old, dep_new], dim=0)

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

# Check parity
ref_mask, _ = cache.compute_invalidation_from_cache(changed_idx, rad_dyn, m2d_dyn, dep_dyn)
jit_mask = fused_invalidation_complete(
    cache.cached_means2d, cache.cached_radii, cache.cached_depths,
    changed_idx, rad_dyn, m2d_dyn, dep_dyn,
    y_grid, x_grid, width, height, tile_size, tile_w, tile_h
)
print("Complete JIT bitwise identical:", torch.equal(ref_mask, jit_mask))

# Warmup
for _ in range(30):
    _ = fused_invalidation_complete(
        cache.cached_means2d, cache.cached_radii, cache.cached_depths,
        changed_idx, rad_dyn, m2d_dyn, dep_dyn,
        y_grid, x_grid, width, height, tile_size, tile_w, tile_h
    )
torch.cuda.synchronize()

reps = 100
starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
for i in range(reps):
    starts[i].record()
    _ = fused_invalidation_complete(
        cache.cached_means2d, cache.cached_radii, cache.cached_depths,
        changed_idx, rad_dyn, m2d_dyn, dep_dyn,
        y_grid, x_grid, width, height, tile_size, tile_w, tile_h
    )
    ends[i].record()
torch.cuda.synchronize()
times = [starts[i].elapsed_time(ends[i]) for i in range(reps)]

print(f"Fused Complete Invalidation: {np.median(times):.4f} ms ± {np.std(times):.4f} ms")
