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

# Method 1: Existing compute_invalidation_from_cache
mask_existing, px_existing = cache.compute_invalidation_from_cache(changed_idx, rad_dyn, m2d_dyn, dep_dyn)

# Method 2: Optimized 2D difference array invalidation
def compute_invalidation_diff_array(dynamic_indices, rad_dyn, m2d_dyn, dep_dyn):
    if dynamic_indices.numel() == 0:
        return torch.zeros((tile_h, tile_w), dtype=torch.bool, device=device)

    # Old from cache + new dynamic
    m2d_all = torch.cat([cache.cached_means2d[0, dynamic_indices], m2d_dyn[0]], dim=0)
    rad_all = torch.cat([cache.cached_radii[0, dynamic_indices], rad_dyn[0]], dim=0)
    dep_all = torch.cat([cache.cached_depths[0, dynamic_indices], dep_dyn[0]], dim=0)

    valid = (dep_all > 0.01) & (rad_all[:, 0] > 0) & (rad_all[:, 1] > 0)
    xy_min = m2d_all - rad_all
    xy_max = m2d_all + rad_all
    in_screen = (xy_max >= 0).all(dim=-1) & (xy_min[:, 0] < width) & (xy_min[:, 1] < height)
    active = valid & in_screen

    # Filter only active
    xy_min = xy_min[active]
    xy_max = xy_max[active]
    if xy_min.numel() == 0:
        return torch.zeros((tile_h, tile_w), dtype=torch.bool, device=device)

    u_min = torch.clamp((xy_min[:, 0] / tile_size).floor().long(), 0, tile_w - 1)
    u_max = torch.clamp((xy_max[:, 0] / tile_size).floor().long(), 0, tile_w - 1)
    v_min = torch.clamp((xy_min[:, 1] / tile_size).floor().long(), 0, tile_h - 1)
    v_max = torch.clamp((xy_max[:, 1] / tile_size).floor().long(), 0, tile_h - 1)

    # 2D difference array: shape (tile_h + 1, tile_w + 1)
    # Pitch is (tile_w + 1)
    pitch = tile_w + 1
    diff = torch.zeros(pitch * (tile_h + 1), dtype=torch.int32, device=device)

    p1 = v_min * pitch + u_min
    p2 = v_min * pitch + (u_max + 1)
    p3 = (v_max + 1) * pitch + u_min
    p4 = (v_max + 1) * pitch + (u_max + 1)

    ones = torch.ones_like(p1, dtype=torch.int32)
    diff.scatter_add_(0, p1, ones)
    diff.scatter_add_(0, p2, -ones)
    diff.scatter_add_(0, p3, -ones)
    diff.scatter_add_(0, p4, ones)

    diff_2d = diff.view(tile_h + 1, pitch)
    prefix_2d = torch.cumsum(torch.cumsum(diff_2d, dim=0), dim=1)
    invalid_tile_mask = (prefix_2d[:tile_h, :tile_w] > 0)
    return invalid_tile_mask

mask_diff = compute_invalidation_diff_array(changed_idx, rad_dyn, m2d_dyn, dep_dyn)

# Correctness check: Exactly identical?
is_identical = torch.equal(mask_existing, mask_diff)
print(f"Masks are bitwise identical: {is_identical}")
if not is_identical:
    print(f"Diff count: {(mask_existing != mask_diff).sum().item()}")

# Benchmark both
reps = 100
for _ in range(20):
    _ = cache.compute_invalidation_from_cache(changed_idx, rad_dyn, m2d_dyn, dep_dyn)
    _ = compute_invalidation_diff_array(changed_idx, rad_dyn, m2d_dyn, dep_dyn)
torch.cuda.synchronize()

def bench(fn):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = [starts[i].elapsed_time(ends[i]) for i in range(reps)]
    return np.median(times), np.std(times)

t_ex_med, t_ex_std = bench(lambda: cache.compute_invalidation_from_cache(changed_idx, rad_dyn, m2d_dyn, dep_dyn))
t_diff_med, t_diff_std = bench(lambda: compute_invalidation_diff_array(changed_idx, rad_dyn, m2d_dyn, dep_dyn))

print(f"Existing Invalidation:    {t_ex_med:.4f} ms ± {t_ex_std:.4f} ms")
print(f"Diff Array Invalidation:  {t_diff_med:.4f} ms ± {t_diff_std:.4f} ms")
print(f"Speedup:                  {t_ex_med / t_diff_med:.2f}x")
