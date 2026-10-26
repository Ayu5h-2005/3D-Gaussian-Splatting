"""
run_milestone_2d_benchmark.py - Milestone 2D Scene Scaling & Sorting Reuse Benchmark

Executes comprehensive scaling and static sorting reuse benchmarks across:
- Scene Scales: 5k, 10k, 25k, 50k, and 100k Gaussians
- Perturbations: Single-Gaussian, Spatial Cluster (1%, 5%, 10%), Random Subset (1%, 5%, 10%)
- Per-stage timing breakdown (Dynamic Proj, Invalidation, Buffer Update, isect_tiles, offset_encode, Masked Rast, Composite)
- Peak GPU memory allocation (MB)
- Image fidelity metrics (PSNR, SSIM, Max Error)
- Zero False Negatives verification (Recall = 1.0)
- Static Tile Sorting Reuse & Candidate-subset sorting comparison
- Saves structured CSV and JSON reports to results/
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

from src.baseline_3dgs import generate_procedural_scene
from src.scene_perturbation import perturb_scene
from src.scene_cache import StaticSceneCache
from src.tile_invalidation import (
    compute_ground_truth_changed_tiles,
    evaluate_invalidation_coverage,
    composite_with_cache,
    compute_image_metrics,
)


def benchmark_stage(fn, warmups: int, reps: int):
    """Benchmarks an isolated callable function using CUDA Events."""
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    t0 = time.perf_counter()
    for i in range(reps):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    times = np.array([starts[i].elapsed_time(ends[i]) for i in range(reps)])
    cpu_wall_ms = ((t1 - t0) * 1000.0) / max(reps, 1)
    return float(np.median(times)), float(np.std(times)), float(cpu_wall_ms)


def run_benchmark():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Milestone 2D benchmark")

    width, height = 512, 512
    tile_size = 16
    tile_w = width // tile_size
    tile_h = height // tile_size
    total_tiles = tile_w * tile_h

    scene_sizes = [5000, 10000, 25000, 50000, 100000]
    warmups = 10
    reps = 30

    print("=" * 125)
    print("MILESTONE 2D: SCENE SCALING & STATIC TILE-SORTING REUSE BENCHMARK")
    print(f"Device: {torch.cuda.get_device_name(device)} | PyTorch: {torch.__version__} | gsplat: {gsplat.__version__}")
    print(f"Resolution: {width}x{height} | Tile Size: {tile_size}x{tile_size} ({total_tiles} tiles)")
    print(f"Scene Scales Tested: {scene_sizes}")
    print(f"Reps per configuration: {reps} (after {warmups} warmups)")
    print("=" * 125)
    print(
        f"{'Scene':<7} | {'Config':<18} | {'Full (ms)':<10} | {'Incr (ms)':<10} | "
        f"{'Speedup':<8} | {'VRAM(MB)':<9} | {'Dirty%':<7} | {'PSNR':<6} | {'SSIM':<6} | {'FN':<3}"
    )
    print("-" * 125)

    all_results = []
    sorting_reuse_analysis = []

    for num_gaussians in scene_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        scene = generate_procedural_scene(num_gaussians=num_gaussians, device=device, seed=42)
        scene["viewmats"] = torch.eye(4, device=device)[None, :, :]
        scene["Ks"] = torch.tensor([[512.0, 0, 256.0], [0, 512.0, 256.0], [0, 0, 1.0]], device=device)[None, :, :]
        scene["width"] = width
        scene["height"] = height

        t_setup_0 = time.perf_counter()
        cache = StaticSceneCache(scene, tile_size=tile_size)
        torch.cuda.synchronize()
        cache_setup_ms = (time.perf_counter() - t_setup_0) * 1000.0

        # Define perturbations for this scene size
        configs = [
            {"mode": "single", "fraction": 1.0 / num_gaussians, "name": "Single Gaussian"},
            {"mode": "spatial_cluster", "fraction": 0.01, "name": "1% Spatial Cluster"},
            {"mode": "spatial_cluster", "fraction": 0.05, "name": "5% Spatial Cluster"},
            {"mode": "spatial_cluster", "fraction": 0.10, "name": "10% Spatial Cluster"},
            {"mode": "random_subset", "fraction": 0.01, "name": "1% Random Subset"},
            {"mode": "random_subset", "fraction": 0.05, "name": "5% Random Subset"},
            {"mode": "random_subset", "fraction": 0.10, "name": "10% Random Subset"},
        ]

        # Candidate vs Global Sorting Profile for 1% cluster
        perturbed_probe, changed_idx_probe, _ = perturb_scene(
            scene, fraction=0.01, delta=(0.15, -0.10, 0.05), seed=42, mode="spatial_cluster"
        )
        rad_dp, m2d_dp, dep_dp, con_dp = cache.project_dynamic_subset(perturbed_probe, changed_idx_probe)
        inv_mp, _ = cache.compute_invalidation_from_cache(changed_idx_probe, rad_dp, m2d_dp, dep_dp, generate_pixel_mask=False)
        cache.update_working_buffers(changed_idx_probe, rad_dp, m2d_dp, dep_dp, con_dp)

        # Profile Global RadixSort
        t_glob_sort_med, _, _ = benchmark_stage(
            lambda: gsplat.isect_tiles(cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h),
            warmups, reps
        )
        _, isect_probe, _ = gsplat.isect_tiles(cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h)
        t_glob_offset_med, _, _ = benchmark_stage(
            lambda: gsplat.isect_offset_encode(isect_probe, 1, tile_w, tile_h),
            warmups, reps
        )

        # Profile Candidate Filtering & Candidate RadixSort
        t_cand_filt_med, _, _ = benchmark_stage(
            lambda: cache.filter_candidate_gaussians(inv_mp),
            warmups, reps
        )
        c_idx_probe = cache.filter_candidate_gaussians(inv_mp)
        n_cand = c_idx_probe.numel()
        m_s = cache.buf_means2d[0, c_idx_probe].contiguous()
        r_s = cache.buf_radii[0, c_idx_probe].contiguous()
        d_s = cache.buf_depths[0, c_idx_probe].contiguous()
        im_ids = torch.zeros(n_cand, device=device, dtype=torch.int64)

        t_cand_sort_med, _, _ = benchmark_stage(
            lambda: gsplat.isect_tiles(m_s, r_s, d_s, tile_size, tile_w, tile_h, sort=True, packed=True, n_images=1, image_ids=im_ids, gaussian_ids=c_idx_probe),
            warmups, reps
        )
        _, isect_cand, flat_cand = gsplat.isect_tiles(m_s, r_s, d_s, tile_size, tile_w, tile_h, sort=True, packed=True, n_images=1, image_ids=im_ids, gaussian_ids=c_idx_probe)
        t_cand_offset_med, _, _ = benchmark_stage(
            lambda: gsplat.isect_offset_encode(isect_cand, 1, tile_w, tile_h),
            warmups, reps
        )
        t_cand_map_med, _, _ = benchmark_stage(
            lambda: c_idx_probe[flat_cand.long()].int(),
            warmups, reps
        )
        cache.reset_working_buffers_to_cache(changed_idx_probe)

        sorting_reuse_analysis.append({
            "num_gaussians": num_gaussians,
            "candidate_count": n_cand,
            "candidate_ratio": round(n_cand / num_gaussians, 4),
            "t_global_sort_ms": round(t_glob_sort_med, 4),
            "t_global_offset_ms": round(t_glob_offset_med, 4),
            "t_candidate_filter_ms": round(t_cand_filt_med, 4),
            "t_candidate_sort_ms": round(t_cand_sort_med, 4),
            "t_candidate_offset_ms": round(t_cand_offset_med, 4),
            "t_candidate_index_map_ms": round(t_cand_map_med, 4),
            "candidate_total_overhead_ms": round(t_cand_filt_med + t_cand_sort_med + t_cand_offset_med + t_cand_map_med, 4),
            "global_total_overhead_ms": round(t_glob_sort_med + t_glob_offset_med, 4),
        })

        # Run benchmarks for all configurations of this scene size
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

            # Ground truth full render
            ref_colors, _, _ = gsplat.rasterization(
                perturbed["means"], perturbed["quats"], perturbed["scales"],
                perturbed["opacities"], perturbed["colors"],
                perturbed["viewmats"], perturbed["Ks"], width, height
            )
            img_ref = ref_colors[0]

            # Evaluate invalidation coverage against ground truth changed tiles
            gt_tiles = compute_ground_truth_changed_tiles(cache.cached_frame, img_ref, tile_size=tile_size)
            gt_count = int(gt_tiles.sum().item())

            rad_d, m2d_d, dep_d, con_d = cache.project_dynamic_subset(perturbed, changed_idx)
            inv_m, _ = cache.compute_invalidation_from_cache(changed_idx, rad_d, m2d_d, dep_d, generate_pixel_mask=False)
            cov_eval = evaluate_invalidation_coverage(inv_m, gt_tiles)

            # Test incremental render for mathematical parity
            cache.update_working_buffers(changed_idx, rad_d, m2d_d, dep_d, con_d)
            _, iscts, flts = gsplat.isect_tiles(cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h)
            ofst = gsplat.isect_offset_encode(iscts, 1, tile_w, tile_h)
            sparse_cols, _ = gsplat.rasterize_to_pixels(
                cache.buf_means2d, cache.buf_conics, pert_colors, pert_opacities,
                width, height, tile_size, ofst, flts, masks=inv_m[None, ...]
            )
            comp_img = cache.composite(sparse_cols[0], inv_m)
            cache.reset_working_buffers_to_cache(changed_idx)

            img_metrics = compute_image_metrics(img_ref, comp_img)

            # Timing: Per-stage isolated benchmarks
            t_dyn_proj, _, _ = benchmark_stage(
                lambda: cache.project_dynamic_subset(perturbed, changed_idx), warmups, reps
            )
            t_inval, _, _ = benchmark_stage(
                lambda: cache.compute_invalidation_from_cache(changed_idx, rad_d, m2d_d, dep_d, generate_pixel_mask=False),
                warmups, reps
            )
            t_buf_update, _, _ = benchmark_stage(
                lambda: cache.update_working_buffers(changed_idx, rad_d, m2d_d, dep_d, con_d),
                warmups, reps
            )
            t_isect_sort, _, _ = benchmark_stage(
                lambda: gsplat.isect_tiles(cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h),
                warmups, reps
            )
            t_offset, _, _ = benchmark_stage(
                lambda: gsplat.isect_offset_encode(iscts, 1, tile_w, tile_h),
                warmups, reps
            )
            t_raster_masked, _, _ = benchmark_stage(
                lambda: gsplat.rasterize_to_pixels(
                    cache.buf_means2d, cache.buf_conics, pert_colors, pert_opacities,
                    width, height, tile_size, ofst, flts, masks=inv_m[None, ...]
                ),
                warmups, reps
            )
            t_raster_full, _, _ = benchmark_stage(
                lambda: gsplat.rasterize_to_pixels(
                    cache.buf_means2d, cache.buf_conics, pert_colors, pert_opacities,
                    width, height, tile_size, ofst, flts
                ),
                warmups, reps
            )
            t_composite, _, _ = benchmark_stage(
                lambda: cache.composite(sparse_cols[0], inv_m),
                warmups, reps
            )

            # Timing: Chained Full Rerender
            t_full_med, t_full_std, c_full = benchmark_stage(
                lambda: gsplat.rasterization(
                    perturbed["means"], perturbed["quats"], perturbed["scales"],
                    perturbed["opacities"], perturbed["colors"],
                    perturbed["viewmats"], perturbed["Ks"], width, height
                ),
                warmups, reps
            )

            # Timing: Chained Incremental Pipeline (Steady-state)
            def run_incremental():
                r, m, d, c = cache.project_dynamic_subset(perturbed, changed_idx)
                m_inv, _ = cache.compute_invalidation_from_cache(changed_idx, r, m, d, generate_pixel_mask=False)
                cache.update_working_buffers(changed_idx, r, m, d, c)
                _, isect, flat = gsplat.isect_tiles(cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h)
                off = gsplat.isect_offset_encode(isect, 1, tile_w, tile_h)
                out_c, _ = gsplat.rasterize_to_pixels(
                    cache.buf_means2d, cache.buf_conics, pert_colors, pert_opacities,
                    width, height, tile_size, off, flat, masks=m_inv[None, ...]
                )
                return cache.composite(out_c[0], m_inv)

            t_incr_med, t_incr_std, c_incr = benchmark_stage(run_incremental, warmups, reps)
            cache.reset_working_buffers_to_cache(changed_idx)

            dirty_count = int(inv_m.sum().item())
            dirty_pct = (dirty_count / total_tiles) * 100.0
            speedup = float(t_full_med / t_incr_med) if t_incr_med > 0 else 1.0
            peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))

            row = {
                "num_gaussians": num_gaussians,
                "config_name": name,
                "mode": mode,
                "fraction": frac,
                "num_changed_gaussians": num_changed,
                "dirty_tiles": dirty_count,
                "dirty_tile_pct": round(dirty_pct, 2),
                "gt_changed_tiles": gt_count,
                "false_negatives": cov_eval["false_negatives"],
                "recall": round(cov_eval["recall"], 4),
                "t_full_render_med_ms": round(t_full_med, 4),
                "t_full_render_std_ms": round(t_full_std, 4),
                "t_full_render_cpu_ms": round(c_full, 4),
                "t_incremental_med_ms": round(t_incr_med, 4),
                "t_incremental_std_ms": round(t_incr_std, 4),
                "t_incremental_cpu_ms": round(c_incr, 4),
                "speedup_ratio": round(speedup, 2),
                "t_stage_dyn_proj_ms": round(t_dyn_proj, 4),
                "t_stage_inval_ms": round(t_inval, 4),
                "t_stage_buf_update_ms": round(t_buf_update, 4),
                "t_stage_isect_sort_ms": round(t_isect_sort, 4),
                "t_stage_offset_ms": round(t_offset, 4),
                "t_stage_raster_masked_ms": round(t_raster_masked, 4),
                "t_stage_raster_full_ms": round(t_raster_full, 4),
                "t_stage_composite_ms": round(t_composite, 4),
                "peak_vram_mb": round(peak_vram_mb, 2),
                "max_abs_error": round(float(img_metrics["max_absolute_error"]), 8),
                "psnr_db": round(float(img_metrics["psnr_db"]), 2) if not math.isinf(img_metrics["psnr_db"]) else "inf",
                "ssim": round(float(img_metrics["ssim"]), 6),
            }
            all_results.append(row)

            psnr_display = str(row["psnr_db"])
            print(
                f"{num_gaussians:<7} | {name:<18} | {t_full_med:<10.3f} | {t_incr_med:<10.3f} | "
                f"{speedup:<7.2f}x | {peak_vram_mb:<9.1f} | {dirty_pct:<6.1f}% | {psnr_display:<6} | {row['ssim']:<6.4f} | {row['false_negatives']:<3}"
            )

    # Save to JSON
    json_path = ROOT / "results" / "milestone_2d_benchmark.json"
    payload = {
        "metadata": {
            "device": torch.cuda.get_device_name(device),
            "pytorch_version": torch.__version__,
            "gsplat_version": gsplat.__version__,
            "resolution": f"{width}x{height}",
            "tile_size": tile_size,
            "total_tiles": total_tiles,
            "scene_scales": scene_sizes,
            "warmups": warmups,
            "reps": reps,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "sorting_reuse_analysis": sorting_reuse_analysis,
        "benchmarks": all_results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Save to CSV
    csv_path = ROOT / "results" / "milestone_2d_benchmark.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)

    print("=" * 125)
    print(f"Milestone 2D Benchmark Complete!")
    print(f"Results saved to:\n  - {json_path}\n  - {csv_path}")
    print("=" * 125)


if __name__ == "__main__":
    run_benchmark()
