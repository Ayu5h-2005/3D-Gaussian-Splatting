"""
run_milestone_2c_benchmark.py - Milestone 2C Comprehensive Benchmarking Suite

Executes rigorous benchmarks comparing Full Baseline Rerender vs Incremental Rendering across:
- Changed Fractions: Single-Gaussian, 1%, 2%, 5%, 10%, 25%
- Perturbation Modes: Spatial Cluster (localized) & Random Subset (distributed)
- 15 warmups, 50 repeated measurements per configuration
- Measures median, standard deviation, and CPU wall-clock times
- Records cache setup time separately from steady-state update latency
- Validates mathematical parity and zero false negatives
- Outputs structured CSV and JSON reports to results/
"""

import csv
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import gsplat

from src.scene_perturbation import perturb_scene
from src.scene_cache import StaticSceneCache
from src.tile_invalidation import (
    compute_ground_truth_changed_tiles,
    evaluate_invalidation_coverage,
    composite_with_cache,
    compute_image_metrics,
)


def run_benchmark():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required for Milestone 2C benchmark")

    scene_path = ROOT / "results" / "procedural_scene_5k.pt"
    scene = torch.load(str(scene_path), weights_only=False)
    for k, v in scene.items():
        if isinstance(v, torch.Tensor):
            scene[k] = v.to(device)

    width = int(scene["width"])
    height = int(scene["height"])
    tile_size = 16
    tile_w = width // tile_size
    tile_h = height // tile_size
    total_tiles = tile_w * tile_h
    num_total_gaussians = scene["means"].shape[0]

    # One-time Cache Setup
    t_setup_0 = time.perf_counter()
    cache = StaticSceneCache(scene, tile_size=tile_size)
    torch.cuda.synchronize()
    cache_setup_ms = (time.perf_counter() - t_setup_0) * 1000.0

    configs = [
        {"mode": "single", "fraction": 0.0002, "name": "Single Gaussian (1 g)"},
        {"mode": "spatial_cluster", "fraction": 0.01, "name": "1% Spatial Cluster (50 g)"},
        {"mode": "spatial_cluster", "fraction": 0.02, "name": "2% Spatial Cluster (100 g)"},
        {"mode": "spatial_cluster", "fraction": 0.05, "name": "5% Spatial Cluster (250 g)"},
        {"mode": "spatial_cluster", "fraction": 0.10, "name": "10% Spatial Cluster (500 g)"},
        {"mode": "spatial_cluster", "fraction": 0.25, "name": "25% Spatial Cluster (1250 g)"},
        {"mode": "random_subset", "fraction": 0.01, "name": "1% Random Subset (50 g)"},
        {"mode": "random_subset", "fraction": 0.02, "name": "2% Random Subset (100 g)"},
        {"mode": "random_subset", "fraction": 0.05, "name": "5% Random Subset (250 g)"},
        {"mode": "random_subset", "fraction": 0.10, "name": "10% Random Subset (500 g)"},
        {"mode": "random_subset", "fraction": 0.25, "name": "25% Random Subset (1250 g)"},
    ]

    warmups = 15
    reps = 50

    results = []

    print("=" * 110)
    print("MILESTONE 2C: COMPREHENSIVE BENCHMARK RUNNER")
    print(f"Device: {torch.cuda.get_device_name(device)} | PyTorch: {torch.__version__} | gsplat: {gsplat.__version__}")
    print(f"Scene: {num_total_gaussians} Gaussians @ {width}x{height} | Tile Size: {tile_size}x{tile_size} ({total_tiles} tiles)")
    print(f"One-Time Cache Setup Latency: {cache_setup_ms:.3f} ms")
    print(f"Evaluation Reps: {reps} (after {warmups} warmups)")
    print("=" * 110)
    print(f"{'Config Name':<28} | {'Dirty%':<7} | {'Full (ms)':<10} | {'Steady (ms)':<11} | {'Incr (ms)':<10} | {'Ratio':<6} | {'PSNR':<6} | {'FN':<3}")
    print("-" * 110)

    for cfg in configs:
        mode = cfg["mode"]
        frac = cfg["fraction"]
        name = cfg["name"]

        perturbed, changed_idx, _ = perturb_scene(
            scene, fraction=frac, delta=(0.15, -0.10, 0.05), seed=42, mode=mode
        )
        num_changed = changed_idx.numel()
        pert_colors = perturbed["colors"][None, ...]
        pert_opacities = perturbed["opacities"][None, ...]

        # Reference Full Rerender
        ref_colors, _, _ = gsplat.rasterization(
            perturbed["means"], perturbed["quats"], perturbed["scales"],
            perturbed["opacities"], perturbed["colors"],
            perturbed["viewmats"], perturbed["Ks"], width, height
        )
        img_ref = ref_colors[0]

        # Ground-truth changed tiles
        gt_tiles = compute_ground_truth_changed_tiles(cache.cached_frame, img_ref, tile_size=tile_size)
        gt_count = int(gt_tiles.sum().item())

        # Correctness & Parity Verification
        rad_d, m2d_d, dep_d, con_d = cache.project_dynamic_subset(perturbed, changed_idx)
        inv_m, _ = cache.compute_invalidation_from_cache(changed_idx, rad_d, m2d_d, dep_d, generate_pixel_mask=False)
        cov_eval = evaluate_invalidation_coverage(inv_m, gt_tiles)

        cache.update_working_buffers(changed_idx, rad_d, m2d_d, dep_d, con_d)
        _, iscts, flts = gsplat.isect_tiles(cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h)
        ofst = gsplat.isect_offset_encode(iscts, 1, tile_w, tile_h)
        sparse_cols, _ = gsplat.rasterize_to_pixels(
            cache.buf_means2d, cache.buf_conics, pert_colors, pert_opacities,
            width, height, tile_size, ofst, flts, masks=inv_m[None, ...]
        )
        comp_img = composite_with_cache(cache.cached_frame, sparse_cols[0], inv_m, tile_size=tile_size)
        cache.reset_working_buffers_to_cache(changed_idx)

        img_metrics = compute_image_metrics(img_ref, comp_img)

        # Execution closures for timing
        def run_full():
            return gsplat.rasterization(
                perturbed["means"], perturbed["quats"], perturbed["scales"],
                perturbed["opacities"], perturbed["colors"],
                perturbed["viewmats"], perturbed["Ks"], width, height
            )

        def run_steady_incremental():
            r, m, d, c = cache.project_dynamic_subset(perturbed, changed_idx)
            m_inv, _ = cache.compute_invalidation_from_cache(changed_idx, r, m, d, generate_pixel_mask=False)
            cache.update_working_buffers(changed_idx, r, m, d, c)
            _, isect, flat = gsplat.isect_tiles(cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h)
            off = gsplat.isect_offset_encode(isect, 1, tile_w, tile_h)
            out_c, _ = gsplat.rasterize_to_pixels(
                cache.buf_means2d, cache.buf_conics, pert_colors, pert_opacities,
                width, height, tile_size, off, flat, masks=m_inv[None, ...]
            )
            return composite_with_cache(cache.cached_frame, out_c[0], m_inv, tile_size=tile_size)

        def run_bench_incremental():
            res = run_steady_incremental()
            cache.reset_working_buffers_to_cache(changed_idx)
            return res

        # Warmup
        for _ in range(warmups):
            _ = run_full()
            _ = run_steady_incremental()
            cache.reset_working_buffers_to_cache(changed_idx)
        torch.cuda.synchronize()

        # Benchmark Full Rerender
        starts_f = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ends_f = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        t0 = time.perf_counter()
        for i in range(reps):
            starts_f[i].record()
            _ = run_full()
            ends_f[i].record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_f = np.array([starts_f[i].elapsed_time(ends_f[i]) for i in range(reps)])
        c_full = ((t1 - t0) * 1000.0) / reps

        # Benchmark Steady-State Incremental (no buffer reset)
        starts_s = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ends_s = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        t0 = time.perf_counter()
        for i in range(reps):
            starts_s[i].record()
            _ = run_steady_incremental()
            ends_s[i].record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_s = np.array([starts_s[i].elapsed_time(ends_s[i]) for i in range(reps)])
        c_steady = ((t1 - t0) * 1000.0) / reps

        # Benchmark Incremental with Buffer Reset (Benchmark Isolation)
        starts_b = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ends_b = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        t0 = time.perf_counter()
        for i in range(reps):
            starts_b[i].record()
            _ = run_bench_incremental()
            ends_b[i].record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_b = np.array([starts_b[i].elapsed_time(ends_b[i]) for i in range(reps)])
        c_bench = ((t1 - t0) * 1000.0) / reps

        dirty_count = int(inv_m.sum().item())
        dirty_pct = (dirty_count / total_tiles) * 100.0
        ratio = float(np.median(times_f) / np.median(times_s))

        row = {
            "name": name,
            "mode": mode,
            "fraction": frac,
            "num_changed_gaussians": num_changed,
            "dirty_tiles": dirty_count,
            "dirty_tile_pct": round(dirty_pct, 2),
            "gt_changed_tiles": gt_count,
            "false_negatives": cov_eval["false_negatives"],
            "recall": round(cov_eval["recall"], 4),
            "t_full_render_med_ms": round(float(np.median(times_f)), 4),
            "t_full_render_std_ms": round(float(np.std(times_f)), 4),
            "t_full_render_cpu_wall_ms": round(float(c_full), 4),
            "t_steady_incremental_med_ms": round(float(np.median(times_s)), 4),
            "t_steady_incremental_std_ms": round(float(np.std(times_s)), 4),
            "t_steady_cpu_wall_ms": round(float(c_steady), 4),
            "t_bench_incremental_med_ms": round(float(np.median(times_b)), 4),
            "t_bench_incremental_std_ms": round(float(np.std(times_b)), 4),
            "steady_state_speedup_ratio": round(ratio, 2),
            "max_abs_error": round(float(img_metrics["max_absolute_error"]), 8),
            "psnr_db": round(float(img_metrics["psnr_db"]), 2) if not math.isinf(img_metrics["psnr_db"]) else "inf",
            "ssim": round(float(img_metrics["ssim"]), 6),
        }
        results.append(row)

        print(
            f"{name:<28} | {dirty_pct:<6.1f}% | {row['t_full_render_med_ms']:<10.3f} | "
            f"{row['t_steady_incremental_med_ms']:<11.3f} | {row['t_bench_incremental_med_ms']:<10.3f} | "
            f"{ratio:<6.2f} | {row['psnr_db']!s:<6} | {row['false_negatives']:<3}"
        )

    # Save to JSON
    json_path = ROOT / "results" / "milestone_2c_benchmark.json"
    payload = {
        "device": torch.cuda.get_device_name(device),
        "pytorch_version": torch.__version__,
        "gsplat_version": gsplat.__version__,
        "cache_setup_ms": round(cache_setup_ms, 3),
        "total_gaussians": num_total_gaussians,
        "resolution": f"{width}x{height}",
        "tile_size": tile_size,
        "benchmarks": results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Save to CSV
    csv_path = ROOT / "results" / "milestone_2c_benchmark.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("=" * 110)
    print(f"Milestone 2C Benchmarking Complete!")
    print(f"Results saved to:\n  - {json_path}\n  - {csv_path}")
    print("=" * 110)


if __name__ == "__main__":
    run_benchmark()
