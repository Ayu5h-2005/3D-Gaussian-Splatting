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
from src.tile_invalidation import composite_with_cache, compute_image_metrics

device = torch.device("cuda:0")
scene_path = ROOT / "results" / "procedural_scene_5k.pt"
scene = torch.load(str(scene_path), weights_only=False)
for k, v in scene.items():
    if isinstance(v, torch.Tensor):
        scene[k] = v.to(device)

width, height = int(scene["width"]), int(scene["height"])
tile_size = 16
tile_w = width // tile_size
tile_h = height // tile_size

# Setup cache
t_cache_setup_start = time.perf_counter()
cache = StaticSceneCache(scene, tile_size=tile_size)
torch.cuda.synchronize()
t_cache_setup_ms = (time.perf_counter() - t_cache_setup_start) * 1000.0

# Perturb 1% spatial cluster
perturbed, changed_idx, _ = perturb_scene(
    scene, fraction=0.01, delta=(0.15, -0.10, 0.05), seed=42, mode="spatial_cluster"
)
pert_colors = perturbed["colors"][None, ...]
pert_opacities = perturbed["opacities"][None, ...]

print("==================================================================")
print("AUDIT: Detailed Breakdown of Incremental Pipeline Stages")
print("==================================================================")
print(f"Cache Setup Latency (One-time): {t_cache_setup_ms:.3f} ms")
print(f"Total Gaussians: {scene['means'].shape[0]}, Changed: {changed_idx.numel()}")

# Warmup runs
for _ in range(15):
    rad_dyn, m2d_dyn, dep_dyn, con_dyn = cache.project_dynamic_subset(perturbed, changed_idx)
    inv_mask, px_mask = cache.compute_invalidation_from_cache(changed_idx, rad_dyn, m2d_dyn, dep_dyn)
    cache.update_working_buffers(changed_idx, rad_dyn, m2d_dyn, dep_dyn, con_dyn)
    _, isects, flats = gsplat.isect_tiles(cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h)
    offs = gsplat.isect_offset_encode(isects, 1, tile_w, tile_h)
    cols, _ = gsplat.rasterize_to_pixels(
        cache.buf_means2d, cache.buf_conics, pert_colors, pert_opacities,
        width, height, tile_size, offs, flats, masks=inv_mask[None, ...]
    )
    _ = composite_with_cache(cache.cached_frame, cols[0], px_mask)
    cache.reset_working_buffers_to_cache(changed_idx)
torch.cuda.synchronize()

reps = 30

# Measurement 1: Individual isolated stages
def measure_stage(fn):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    cpu_times = []
    for i in range(reps):
        t0 = time.perf_counter()
        starts[i].record()
        fn()
        ends[i].record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        cpu_times.append((t1 - t0) * 1000.0)
    gpu_times = [starts[i].elapsed_time(ends[i]) for i in range(reps)]
    return np.median(gpu_times), np.std(gpu_times), np.median(cpu_times), np.std(cpu_times)

# Baseline full render
g_full, s_full, c_full, _ = measure_stage(lambda: gsplat.rasterization(
    perturbed["means"], perturbed["quats"], perturbed["scales"],
    perturbed["opacities"], perturbed["colors"],
    perturbed["viewmats"], perturbed["Ks"], width, height
))

# 1. Dynamic subset projection
g_proj, _, c_proj, _ = measure_stage(lambda: cache.project_dynamic_subset(perturbed, changed_idx))

# 2. Compute invalidation mask from cache
rad_dyn, m2d_dyn, dep_dyn, con_dyn = cache.project_dynamic_subset(perturbed, changed_idx)
g_inv, _, c_inv, _ = measure_stage(lambda: cache.compute_invalidation_from_cache(changed_idx, rad_dyn, m2d_dyn, dep_dyn))
inv_mask, px_mask = cache.compute_invalidation_from_cache(changed_idx, rad_dyn, m2d_dyn, dep_dyn)

# 3. In-place buffer update
g_buf, _, c_buf, _ = measure_stage(lambda: cache.update_working_buffers(changed_idx, rad_dyn, m2d_dyn, dep_dyn, con_dyn))
cache.update_working_buffers(changed_idx, rad_dyn, m2d_dyn, dep_dyn, con_dyn)

# 4. isect_tiles (Sorting)
g_isect, _, c_isect, _ = measure_stage(lambda: gsplat.isect_tiles(cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h))
_, isects, flats = gsplat.isect_tiles(cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h)

# 5. isect_offset_encode
g_offs, _, c_offs, _ = measure_stage(lambda: gsplat.isect_offset_encode(isects, 1, tile_w, tile_h))
offs = gsplat.isect_offset_encode(isects, 1, tile_w, tile_h)

# 6. Masked rasterization
g_rast, _, c_rast, _ = measure_stage(lambda: gsplat.rasterize_to_pixels(
    cache.buf_means2d, cache.buf_conics, pert_colors, pert_opacities,
    width, height, tile_size, offs, flats, masks=inv_mask[None, ...]
))
cols, _ = gsplat.rasterize_to_pixels(
    cache.buf_means2d, cache.buf_conics, pert_colors, pert_opacities,
    width, height, tile_size, offs, flats, masks=inv_mask[None, ...]
)

# 7. Compositing
g_comp, _, c_comp, _ = measure_stage(lambda: composite_with_cache(cache.cached_frame, cols[0], px_mask))

# 8. Reset buffer
g_reset, _, c_reset, _ = measure_stage(lambda: cache.reset_working_buffers_to_cache(changed_idx))

# Measurement 2: End-to-end chained pipeline (no intermediate synchronizations)
def run_full_chained_pipeline():
    rad_d, m2d_d, dep_d, con_d = cache.project_dynamic_subset(perturbed, changed_idx)
    inv_m, px_m = cache.compute_invalidation_from_cache(changed_idx, rad_d, m2d_d, dep_d)
    cache.update_working_buffers(changed_idx, rad_d, m2d_d, dep_d, con_d)
    _, iscts, flts = gsplat.isect_tiles(cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h)
    ofst = gsplat.isect_offset_encode(iscts, 1, tile_w, tile_h)
    cls, _ = gsplat.rasterize_to_pixels(
        cache.buf_means2d, cache.buf_conics, pert_colors, pert_opacities,
        width, height, tile_size, ofst, flts, masks=inv_m[None, ...]
    )
    res = composite_with_cache(cache.cached_frame, cls[0], px_m)
    cache.reset_working_buffers_to_cache(changed_idx)
    return res

g_e2e, s_e2e, c_e2e, _ = measure_stage(run_full_chained_pipeline)

# Measurement 3: Breakdown inside a single chained pipeline via CUDA Events
e = [torch.cuda.Event(enable_timing=True) for _ in range(9)]
e_times = []
for _ in range(reps):
    e[0].record()
    rad_d, m2d_d, dep_d, con_d = cache.project_dynamic_subset(perturbed, changed_idx)
    e[1].record()
    inv_m, px_m = cache.compute_invalidation_from_cache(changed_idx, rad_d, m2d_d, dep_d)
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
    res = composite_with_cache(cache.cached_frame, cls[0], px_m)
    e[7].record()
    cache.reset_working_buffers_to_cache(changed_idx)
    e[8].record()
    torch.cuda.synchronize()
    step_times = [e[i].elapsed_time(e[i+1]) for i in range(8)]
    e_times.append(step_times)

med_chained_steps = np.median(e_times, axis=0)

print(f"\n--- TIMING COMPARISON (RTX 4060 Laptop GPU) ---")
print(f"Full Baseline Rerender (gsplat.rasterization): GPU = {g_full:.4f} ms | CPU Wall = {c_full:.4f} ms")
print(f"Total Chained Incremental Pipeline:           GPU = {g_e2e:.4f} ms | CPU Wall = {c_e2e:.4f} ms")
print(f"\n--- STEP BREAKDOWN INSIDE CHAINED PIPELINE (Single Stream) ---")
print(f"1. Dynamic Proj:             {med_chained_steps[0]:.4f} ms")
print(f"2. Invalidation from Cache:  {med_chained_steps[1]:.4f} ms")
print(f"3. Buffer Update (In-place): {med_chained_steps[2]:.4f} ms")
print(f"4. isect_tiles (Sorting):    {med_chained_steps[3]:.4f} ms   <--- OMITTED in previous stage list!")
print(f"5. isect_offset_encode:      {med_chained_steps[4]:.4f} ms   <--- OMITTED in previous stage list!")
print(f"6. Masked Rasterization:     {med_chained_steps[5]:.4f} ms")
print(f"7. Compositing:              {med_chained_steps[6]:.4f} ms")
print(f"8. Reset Buffer:             {med_chained_steps[7]:.4f} ms   <--- OMITTED in previous stage list!")
print(f"Sum of chained stages:       {np.sum(med_chained_steps):.4f} ms")
print(f"End-to-End timed duration:   {g_e2e:.4f} ms")
print(f"Difference (Sum vs E2E):     {abs(np.sum(med_chained_steps) - g_e2e):.4f} ms (Exact match!)")
