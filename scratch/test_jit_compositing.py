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
from src.tile_invalidation import composite_with_cache

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
inv_mask, px_mask = cache.compute_invalidation_from_cache(changed_idx, rad_dyn, m2d_dyn, dep_dyn)

sparse_colors = torch.rand_like(cache.cached_frame)

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

res_orig = composite_with_cache(cache.cached_frame, sparse_colors, px_mask)
res_broad = composite_tile_broadcast(cache.cached_frame, sparse_colors, inv_mask, tile_h, tile_w, tile_size)

print("Compositing outputs bitwise identical:", torch.equal(res_orig, res_broad))

# Benchmark
reps = 100
for _ in range(30):
    _ = composite_with_cache(cache.cached_frame, sparse_colors, px_mask)
    _ = composite_tile_broadcast(cache.cached_frame, sparse_colors, inv_mask, tile_h, tile_w, tile_size)
torch.cuda.synchronize()

starts_o = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
ends_o = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
for i in range(reps):
    starts_o[i].record()
    _ = composite_with_cache(cache.cached_frame, sparse_colors, px_mask)
    ends_o[i].record()
torch.cuda.synchronize()

starts_b = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
ends_b = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
for i in range(reps):
    starts_b[i].record()
    _ = composite_tile_broadcast(cache.cached_frame, sparse_colors, inv_mask, tile_h, tile_w, tile_size)
    ends_b[i].record()
torch.cuda.synchronize()

times_o = [starts_o[i].elapsed_time(ends_o[i]) for i in range(reps)]
times_b = [starts_b[i].elapsed_time(ends_b[i]) for i in range(reps)]

print(f"Original composite (with pixel_mask): {np.median(times_o):.4f} ms")
print(f"Broadcast composite (tile_mask only): {np.median(times_b):.4f} ms")
