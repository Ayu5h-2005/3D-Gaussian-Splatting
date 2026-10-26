import sys
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
scene = torch.load(ROOT / "results" / "procedural_scene_5k.pt", weights_only=False)
for k, v in scene.items():
    if isinstance(v, torch.Tensor):
        scene[k] = v.to(device)

width, height = int(scene["width"]), int(scene["height"])
tile_size = 16
tile_w, tile_h = width // tile_size, height // tile_size

# Initialize cache
cache = StaticSceneCache(scene, tile_size=tile_size)
assert cache.is_valid(scene["viewmats"], scene["Ks"], width, height, tile_size)

# Apply 1% spatial cluster
perturbed, changed_idx, _ = perturb_scene(
    scene, fraction=0.01, delta=(0.12, -0.10, 0.04), seed=42, mode="spatial_cluster"
)

# Baseline full rerender
ref_img, _, _ = gsplat.rasterization(
    perturbed["means"], perturbed["quats"], perturbed["scales"],
    perturbed["opacities"], perturbed["colors"],
    perturbed["viewmats"], perturbed["Ks"], width, height
)

# Pipeline A: In-place working buffer (all 5000 Gaussians sorted/rasterized with tile mask)
def run_pipeline_inplace():
    rad_dyn, m2d_dyn, dep_dyn, con_dyn = cache.project_dynamic_subset(perturbed, changed_idx)
    inv_mask, px_mask = cache.compute_invalidation_from_cache(changed_idx, rad_dyn, m2d_dyn, dep_dyn)
    cache.update_working_buffers(changed_idx, rad_dyn, m2d_dyn, dep_dyn, con_dyn)
    
    _, isects, flats = gsplat.isect_tiles(
        cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h
    )
    offs = gsplat.isect_offset_encode(isects, 1, tile_w, tile_h)
    cols, _ = gsplat.rasterize_to_pixels(
        cache.buf_means2d, cache.buf_conics, perturbed["colors"][None, ...], perturbed["opacities"][None, ...],
        width, height, tile_size, offs, flats, masks=inv_mask[None, ...]
    )
    out = composite_with_cache(cache.cached_frame, cols[0], px_mask)
    cache.reset_working_buffers_to_cache(changed_idx)
    return out

# Pipeline B: In-place working buffer + Candidate Filtering (only Gaussians touching dirty tiles)
def run_pipeline_candidate():
    rad_dyn, m2d_dyn, dep_dyn, con_dyn = cache.project_dynamic_subset(perturbed, changed_idx)
    inv_mask, px_mask = cache.compute_invalidation_from_cache(changed_idx, rad_dyn, m2d_dyn, dep_dyn)
    cache.update_working_buffers(changed_idx, rad_dyn, m2d_dyn, dep_dyn, con_dyn)
    
    cand_idx = cache.filter_candidate_gaussians(inv_mask)
    if cand_idx.numel() == 0:
        out = cache.cached_frame
    else:
        m2d_c = cache.buf_means2d[:, cand_idx]
        rad_c = cache.buf_radii[:, cand_idx]
        dep_c = cache.buf_depths[:, cand_idx]
        con_c = cache.buf_conics[:, cand_idx]
        col_c = perturbed["colors"][None, cand_idx]
        opac_c = perturbed["opacities"][None, cand_idx]
        
        _, isects, flats = gsplat.isect_tiles(m2d_c, rad_c, dep_c, tile_size, tile_w, tile_h)
        offs = gsplat.isect_offset_encode(isects, 1, tile_w, tile_h)
        cols, _ = gsplat.rasterize_to_pixels(
            m2d_c, con_c, col_c, opac_c,
            width, height, tile_size, offs, flats, masks=inv_mask[None, ...]
        )
        out = composite_with_cache(cache.cached_frame, cols[0], px_mask)
        
    cache.reset_working_buffers_to_cache(changed_idx)
    return out

# Check parity for both
out_a = run_pipeline_inplace()
m_a = compute_image_metrics(ref_img[0], out_a)
print(f"Pipeline A Parity: MaxError={m_a['max_absolute_error']}, PSNR={m_a['psnr_db']}, SSIM={m_a['ssim']}")

out_b = run_pipeline_candidate()
m_b = compute_image_metrics(ref_img[0], out_b)
print(f"Pipeline B Parity: MaxError={m_b['max_absolute_error']}, PSNR={m_b['psnr_db']}, SSIM={m_b['ssim']}")

# Benchmark 20 warmups, 50 reps
warmups = 20
reps = 50

for _ in range(warmups):
    _ = run_pipeline_inplace()
    _ = run_pipeline_candidate()
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

t_full_med, t_full_std = bench(lambda: gsplat.rasterization(
    perturbed["means"], perturbed["quats"], perturbed["scales"],
    perturbed["opacities"], perturbed["colors"],
    perturbed["viewmats"], perturbed["Ks"], width, height
))

t_a_med, t_a_std = bench(run_pipeline_inplace)
t_b_med, t_b_std = bench(run_pipeline_candidate)

print(f"Baseline Full Render: {t_full_med:.4f} ms ± {t_full_std:.4f} ms")
print(f"Pipeline A (In-place buffer): {t_a_med:.4f} ms ± {t_a_std:.4f} ms")
print(f"Pipeline B (Candidate Filter): {t_b_med:.4f} ms ± {t_b_std:.4f} ms")
