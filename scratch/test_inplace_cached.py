import sys
from pathlib import Path

ROOT = Path(r"D:\3dgs-incremental-research")
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import gsplat
from src.scene_perturbation import perturb_scene
from src.tile_invalidation import composite_with_cache, compute_image_metrics

device = torch.device("cuda:0")
scene = torch.load(ROOT / "results" / "procedural_scene_5k.pt", weights_only=False)
for k, v in scene.items():
    if isinstance(v, torch.Tensor):
        scene[k] = v.to(device)

width, height = int(scene["width"]), int(scene["height"])
tile_size = 16
tile_w, tile_h = width // tile_size, height // tile_size

# Pre-render baseline t0
colors_t0, _, _ = gsplat.rasterization(
    scene["means"], scene["quats"], scene["scales"],
    scene["opacities"], scene["colors"],
    scene["viewmats"], scene["Ks"], width, height
)
cached_frame = colors_t0[0].clone()

# Precompute and cache static projections
rad_cache, m2d_cache, dep_cache, con_cache, _ = gsplat.fully_fused_projection(
    scene["means"], None, scene["quats"], scene["scales"],
    scene["viewmats"], scene["Ks"], width, height, packed=False
)
# Pre-allocate updated buffers so we never allocate in the loop
m2d_buf = m2d_cache.clone()
rad_buf = rad_cache.clone()
dep_buf = dep_cache.clone()
con_buf = con_cache.clone()

# Pre-compute tile grids
y_grid = torch.arange(tile_h, device=device).view(tile_h, 1)
x_grid = torch.arange(tile_w, device=device).view(1, tile_w)

# Apply 1% spatial cluster perturbation
perturbed, changed_idx, _ = perturb_scene(
    scene, fraction=0.01, delta=(0.12, -0.10, 0.04), seed=42, mode="spatial_cluster"
)

# Reference full rerender
img_ref, _, _ = gsplat.rasterization(
    perturbed["means"], perturbed["quats"], perturbed["scales"],
    perturbed["opacities"], perturbed["colors"],
    perturbed["viewmats"], perturbed["Ks"], width, height
)

# Test function
def run_cached_pipeline():
    # 1. Project dynamic Gaussians (50 Gaussians)
    rad_dyn, m2d_dyn, dep_dyn, con_dyn, _ = gsplat.fully_fused_projection(
        perturbed["means"][changed_idx], None,
        perturbed["quats"][changed_idx], perturbed["scales"][changed_idx],
        perturbed["viewmats"], perturbed["Ks"], width, height, packed=False
    )

    # 2. Vectorized Invalidation using cached old & freshly projected new
    # Old footprint
    m2d_o = m2d_cache[0, changed_idx]
    rad_o = rad_cache[0, changed_idx]
    dep_o = dep_cache[0, changed_idx]
    
    # Concatenate old and new dynamic (100 Gaussians total)
    m2d_all = torch.cat([m2d_o, m2d_dyn[0]], dim=0)
    rad_all = torch.cat([rad_o, rad_dyn[0]], dim=0)
    dep_all = torch.cat([dep_o, dep_dyn[0]], dim=0)
    
    valid = (dep_all > 0.01) & (rad_all > 0).all(dim=-1)
    xy_min = m2d_all - rad_all
    xy_max = m2d_all + rad_all
    in_screen = (xy_max >= 0).all(dim=-1) & (xy_min[:, 0] < width) & (xy_min[:, 1] < height)
    active = valid & in_screen
    
    uv_min = torch.clamp((xy_min / tile_size).floor().long(), 0, tile_w - 1)
    uv_max = torch.clamp((xy_max / tile_size).floor().long(), 0, tile_h - 1)
    
    Y = ((y_grid >= uv_min[:, 1].view(1, -1)) & (y_grid <= uv_max[:, 1].view(1, -1))).float() * active.view(1, -1).float()
    X = ((x_grid >= uv_min[:, 0].view(-1, 1)) & (x_grid <= uv_max[:, 0].view(-1, 1))).float()
    invalid_mask = (torch.mm(Y, X) > 0)
    
    # 3. Update 2D buffer in-place
    m2d_buf[0, changed_idx] = m2d_dyn[0]
    rad_buf[0, changed_idx] = rad_dyn[0]
    dep_buf[0, changed_idx] = dep_dyn[0]
    con_buf[0, changed_idx] = con_dyn[0]
    
    # 4. isect_tiles on updated 2D projections
    _, isect_ids, flatten_ids = gsplat.isect_tiles(
        m2d_buf, rad_buf, dep_buf, tile_size, tile_w, tile_h
    )
    offsets = gsplat.isect_offset_encode(isect_ids, 1, tile_w, tile_h)
    
    # 5. Tile-masked rasterization (only dirty tiles)
    sparse_colors, _ = gsplat.rasterize_to_pixels(
        m2d_buf, con_buf, perturbed["colors"][None, ...], perturbed["opacities"][None, ...],
        width, height, tile_size, offsets, flatten_ids,
        masks=invalid_mask[None, ...]
    )
    
    # 6. Compositing
    pixel_mask = invalid_mask.repeat_interleave(tile_size, dim=0).repeat_interleave(tile_size, dim=1)[:height, :width, None]
    comp = composite_with_cache(cached_frame, sparse_colors[0], pixel_mask)
    return comp

# Warmup
for _ in range(15):
    comp = run_cached_pipeline()
torch.cuda.synchronize()

# Verify mathematical correctness
metrics = compute_image_metrics(img_ref[0], comp)
print("Mathematical Parity:")
print(f"  Max Absolute Error: {metrics['max_absolute_error']}")
print(f"  PSNR: {metrics['psnr_db']}")
print(f"  SSIM: {metrics['ssim']}")

# Benchmark Full Rerender Baseline
reps = 50
starts_full = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
ends_full = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
for i in range(reps):
    starts_full[i].record()
    _ = gsplat.rasterization(
        perturbed["means"], perturbed["quats"], perturbed["scales"],
        perturbed["opacities"], perturbed["colors"],
        perturbed["viewmats"], perturbed["Ks"], width, height
    )
    ends_full[i].record()
torch.cuda.synchronize()
times_full = [starts_full[i].elapsed_time(ends_full[i]) for i in range(reps)]
print(f"Baseline Full Rerender: Median = {np.median(times_full):.4f} ms ± {np.std(times_full):.4f} ms")

# Benchmark Cached In-Place Pipeline
starts_cached = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
ends_cached = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
for i in range(reps):
    starts_cached[i].record()
    _ = run_cached_pipeline()
    ends_cached[i].record()
torch.cuda.synchronize()
times_cached = [starts_cached[i].elapsed_time(ends_cached[i]) for i in range(reps)]
print(f"Milestone 2B Cached Pipeline: Median = {np.median(times_cached):.4f} ms ± {np.std(times_cached):.4f} ms")
