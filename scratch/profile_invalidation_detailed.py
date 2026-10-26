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

# Let's break down compute_invalidation_from_cache line by line
reps = 100

evs = [torch.cuda.Event(enable_timing=True) for _ in range(12)]
timings = []

# Warmup
for _ in range(20):
    _ = cache.compute_invalidation_from_cache(changed_idx, rad_dyn, m2d_dyn, dep_dyn)
torch.cuda.synchronize()

for _ in range(reps):
    evs[0].record()
    # Step 1: Slicing old footprints from cache
    m2d_old = cache.cached_means2d[0, changed_idx]
    rad_old = cache.cached_radii[0, changed_idx]
    dep_old = cache.cached_depths[0, changed_idx]
    evs[1].record()

    # Step 2: Unpacking new footprints
    m2d_new = m2d_dyn[0]
    rad_new = rad_dyn[0]
    dep_new = dep_dyn[0]
    evs[2].record()

    # Step 3: Concatenating old + new (2K tensors)
    m2d_all = torch.cat([m2d_old, m2d_new], dim=0)
    rad_all = torch.cat([rad_old, rad_new], dim=0)
    dep_all = torch.cat([dep_old, dep_new], dim=0)
    evs[3].record()

    # Step 4: Validity & bounds
    valid = (dep_all > 0.01) & (rad_all > 0).all(dim=-1)
    xy_min = m2d_all - rad_all
    xy_max = m2d_all + rad_all
    in_screen = (xy_max >= 0).all(dim=-1) & (xy_min[:, 0] < width) & (xy_min[:, 1] < height)
    active = valid & in_screen
    evs[4].record()

    # Step 5: Clamp tile bounds
    uv_min = torch.clamp((xy_min / tile_size).floor().long(), 0, tile_w - 1)
    uv_max = torch.clamp((xy_max / tile_size).floor().long(), 0, tile_h - 1)
    evs[5].record()

    # Step 6: Y grid broadcasting & float conversion
    Y = ((cache.y_grid >= uv_min[:, 1].view(1, -1)) & (cache.y_grid <= uv_max[:, 1].view(1, -1))).float() * active.view(1, -1).float()
    evs[6].record()

    # Step 7: X grid broadcasting & float conversion
    X = ((cache.x_grid >= uv_min[:, 0].view(-1, 1)) & (cache.x_grid <= uv_max[:, 0].view(-1, 1))).float()
    evs[7].record()

    # Step 8: torch.mm matrix multiplication
    invalid_tile_mask = (torch.mm(Y, X) > 0)
    evs[8].record()

    # Step 9: repeat_interleave (creating pixel mask)
    invalid_pixel_mask = (
        invalid_tile_mask.repeat_interleave(tile_size, dim=0)
        .repeat_interleave(tile_size, dim=1)[:height, :width, None]
    )
    evs[9].record()

    torch.cuda.synchronize()
    step_t = [evs[i].elapsed_time(evs[i+1]) for i in range(9)]
    timings.append(step_t)

meds = np.median(timings, axis=0)
labels = [
    "1. Slice old cache (means, rad, dep)",
    "2. Unpack new",
    "3. torch.cat old + new (2K elements)",
    "4. Bounds & screen filter",
    "5. Floor & clamp to tile coords",
    "6. Y broadcast & float conversion (32, 2K)",
    "7. X broadcast & float conversion (2K, 32)",
    "8. torch.mm(Y, X) matrix multiplication",
    "9. repeat_interleave to 512x512x1 pixel mask",
]

print("=== DETAILED LINE-BY-LINE PROFILE OF TILE INVALIDATION ===")
for i in range(9):
    print(f"{labels[i]:<48}: {meds[i]:.4f} ms")
print(f"{'TOTAL INVALIDATION TIME':<48}: {np.sum(meds):.4f} ms")
