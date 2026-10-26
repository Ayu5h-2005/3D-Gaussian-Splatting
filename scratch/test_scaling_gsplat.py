import sys
import time
from pathlib import Path

ROOT = Path(r"D:\3dgs-incremental-research")
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import gsplat
from src.baseline_3dgs import generate_procedural_scene

device = torch.device("cuda:0")
width, height = 512, 512
tile_size = 16
tile_w = width // tile_size
tile_h = height // tile_size

print(f"{'N':<8} | {'Proj (ms)':<10} | {'isect_tiles (ms)':<18} | {'Encode (ms)':<12} | {'Rast (ms)':<10} | {'Total Full (ms)':<16} | {'Peak VRAM (MB)':<14}")
print("-" * 100)

for N in [5000, 10000, 25000, 50000, 100000]:
    torch.cuda.reset_peak_memory_stats()
    scene = generate_procedural_scene(num_gaussians=N, device=device, seed=42)
    viewmats = torch.eye(4, device=device)[None, :, :]
    Ks = torch.tensor([[512.0, 0, 256.0], [0, 512.0, 256.0], [0, 0, 1.0]], device=device)[None, :, :]
    
    # Warmup
    for _ in range(5):
        _ = gsplat.rasterization(
            scene["means"], scene["quats"], scene["scales"],
            scene["opacities"], scene["colors"],
            viewmats, Ks, width, height
        )
    torch.cuda.synchronize()

    reps = 30
    # Measure total full render
    starts_tot = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends_tot = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts_tot[i].record()
        _ = gsplat.rasterization(
            scene["means"], scene["quats"], scene["scales"],
            scene["opacities"], scene["colors"],
            viewmats, Ks, width, height
        )
        ends_tot[i].record()
    torch.cuda.synchronize()
    t_tot = np.median([starts_tot[i].elapsed_time(ends_tot[i]) for i in range(reps)])

    # Breakdown stages
    # 1. Proj
    starts_p = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends_p = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts_p[i].record()
        rad, m2d, dep, con, _ = gsplat.fully_fused_projection(
            scene["means"], None, scene["quats"], scene["scales"],
            viewmats, Ks, width, height, packed=False
        )
        ends_p[i].record()
    torch.cuda.synchronize()
    t_p = np.median([starts_p[i].elapsed_time(ends_p[i]) for i in range(reps)])

    # 2. isect_tiles
    starts_is = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends_is = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts_is[i].record()
        _, isect_ids, flatten_ids = gsplat.isect_tiles(m2d, rad, dep, tile_size, tile_w, tile_h)
        ends_is[i].record()
    torch.cuda.synchronize()
    t_is = np.median([starts_is[i].elapsed_time(ends_is[i]) for i in range(reps)])

    # 3. isect_offset_encode
    starts_off = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends_off = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts_off[i].record()
        offsets = gsplat.isect_offset_encode(isect_ids, 1, tile_w, tile_h)
        ends_off[i].record()
    torch.cuda.synchronize()
    t_off = np.median([starts_off[i].elapsed_time(ends_off[i]) for i in range(reps)])

    # 4. rasterize_to_pixels
    cols_b = scene["colors"][None, ...]
    opac_b = scene["opacities"][None, ...]
    starts_r = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends_r = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts_r[i].record()
        out_img, _ = gsplat.rasterize_to_pixels(
            m2d, con, cols_b, opac_b, width, height, tile_size, offsets, flatten_ids
        )
        ends_r[i].record()
    torch.cuda.synchronize()
    t_r = np.median([starts_r[i].elapsed_time(ends_r[i]) for i in range(reps)])

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    print(f"{N:<8} | {t_p:<10.3f} | {t_is:<18.3f} | {t_off:<12.3f} | {t_r:<10.3f} | {t_tot:<16.3f} | {peak_vram_mb:<14.2f}")
