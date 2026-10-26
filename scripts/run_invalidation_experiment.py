"""
run_invalidation_experiment.py - Reproducible 3DGS Invalidation & Incremental Benchmark (Milestone 2A)

Benchmarks:
1. Ground-truth changed tiles vs predicted invalidation tiles (Zero False Negatives verification).
2. Per-stage execution latency distributions with warmups and repeated measurements:
   - Full Projection (gsplat.fully_fused_projection)
   - Radix Tile Intersection & Sorting (gsplat.isect_tiles)
   - Tile Offset Encoding (gsplat.isect_offset_encode)
   - Full Rasterization Kernel (gsplat.rasterize_to_pixels, masks=None)
   - GPU-Vectorized Invalidation (compute_tile_invalidation_mask)
   - Tile-Masked Rasterization Kernel (gsplat.rasterize_to_pixels, masks=invalid_mask)
   - Pixel Cache Compositing (composite_with_cache)
   - Total End-to-End Full Rerender (gsplat.rasterization)
   - Total End-to-End Incremental Pipeline (Invalidation + Upstream + Masked Raster + Composite)
3. Image reconstruction fidelity: PSNR, SSIM, Max Absolute Difference vs Full Reference Rerender.

Saves:
- CSV: results/invalidation_experiment.csv
- JSON: results/invalidation_experiment.json
- Visualization artifacts in results/
"""

import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image
import gsplat

from src.scene_perturbation import perturb_scene
from src.tile_invalidation import (
    compute_tile_invalidation_mask,
    compute_stats_from_mask,
    compute_ground_truth_changed_tiles,
    evaluate_invalidation_coverage,
    composite_with_cache,
    compute_image_metrics,
)


def measure_stage_distribution(
    fn: Callable[[], any],
    warmup_reps: int = 15,
    timing_reps: int = 50,
) -> Dict[str, any]:
    """
    Executes a callable with warmups and measures per-repetition GPU times using CUDA events.
    Returns distribution dictionary: mean, median, std, p25, p75, iqr, and raw times.
    """
    # 1. Dedicated Warmup
    for _ in range(warmup_reps):
        fn()
    torch.cuda.synchronize()

    # 2. Timing loop using event pairs to isolate per-rep execution without CPU host stalls
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(timing_reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(timing_reps)]

    for i in range(timing_reps):
        starts[i].record()
        fn()
        ends[i].record()

    torch.cuda.synchronize()
    times = np.array([starts[i].elapsed_time(ends[i]) for i in range(timing_reps)], dtype=np.float64)

    mean = float(np.mean(times))
    median = float(np.median(times))
    std = float(np.std(times))
    p25 = float(np.percentile(times, 25))
    p75 = float(np.percentile(times, 75))
    iqr = float(p75 - p25)

    return {
        "mean": mean,
        "median": median,
        "std": std,
        "p25": p25,
        "p75": p75,
        "iqr": iqr,
        "times": times.tolist(),
    }


def run_benchmark(
    scene_path: Path = ROOT / "results" / "procedural_scene_5k.pt",
    output_dir: Path = ROOT / "results",
    warmup_reps: int = 15,
    timing_reps: int = 50,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU is required for this experiment.")

    gpu_name = torch.cuda.get_device_name(device)
    print("=========================================================================================")
    print("Milestone 2A: 3DGS Vectorized Invalidation & Fair Pipeline Benchmark")
    print("=========================================================================================")
    print(f"Device:           {gpu_name}")
    print(f"PyTorch Version:  {torch.__version__}")
    print(f"gsplat Version:   {gsplat.__version__}")
    print(f"Warmup Reps:      {warmup_reps} | Timing Reps: {timing_reps}")
    print("=========================================================================================")

    # Load baseline scene
    scene = torch.load(str(scene_path), weights_only=False)
    for k, v in scene.items():
        if isinstance(v, torch.Tensor):
            scene[k] = v.to(device)

    width = int(scene["width"])
    height = int(scene["height"])
    tile_size = 16
    tile_w = (width + tile_size - 1) // tile_size
    tile_h = (height + tile_size - 1) // tile_size
    total_tiles = tile_w * tile_h
    num_total_gaussians = scene["means"].shape[0]

    # Baseline render at t0
    colors_t0, _, _ = gsplat.rasterization(
        scene["means"], scene["quats"], scene["scales"], scene["opacities"], scene["colors"],
        scene["viewmats"], scene["Ks"], width, height
    )
    cached_frame = colors_t0[0].clone()

    # Experimental conditions
    conditions = [
        {
            "id": "exp0_unchanged",
            "name": "Unchanged Scene (Control)",
            "fraction": 0.0,
            "delta": (0.0, 0.0, 0.0),
            "mode": "random_subset",
            "seed": 42,
        },
        {
            "id": "exp1_single_gaussian",
            "name": "Single Gaussian (0.02%)",
            "fraction": 0.0,
            "delta": (0.10, -0.08, 0.02),
            "mode": "single",
            "seed": 42,
        },
        {
            "id": "exp2_cluster_1pct",
            "name": "1% Spatial Cluster (50 GS)",
            "fraction": 0.01,
            "delta": (0.12, -0.10, 0.04),
            "mode": "spatial_cluster",
            "seed": 42,
        },
        {
            "id": "exp3_random_1pct",
            "name": "1% Random Subset (50 GS)",
            "fraction": 0.01,
            "delta": (0.15, -0.10, 0.05),
            "mode": "random_subset",
            "seed": 42,
        },
        {
            "id": "exp4_random_2pct",
            "name": "2% Random Subset (100 GS)",
            "fraction": 0.02,
            "delta": (0.15, -0.10, 0.05),
            "mode": "random_subset",
            "seed": 42,
        },
        {
            "id": "exp5_random_5pct",
            "name": "5% Random Subset (250 GS)",
            "fraction": 0.05,
            "delta": (0.15, -0.10, 0.05),
            "mode": "random_subset",
            "seed": 42,
        },
        {
            "id": "exp6_boundary_cross",
            "name": "Boundary Crossing (Offscreen)",
            "fraction": 0.01,
            "delta": (2.20, -1.80, 0.0),
            "mode": "random_subset",
            "seed": 99,
        },
    ]

    csv_rows = []
    json_results = []

    for cond in conditions:
        print(f"\n---> Running Condition: {cond['name']} ...")

        # 1. Apply perturbation
        if cond["fraction"] == 0.0 and cond["mode"] != "single":
            perturbed_scene = scene
            changed_idx = torch.tensor([], dtype=torch.long, device=device)
        else:
            perturbed_scene, changed_idx, _ = perturb_scene(
                scene,
                fraction=cond["fraction"],
                delta=cond["delta"],
                seed=cond["seed"],
                mode=cond["mode"],
            )

        num_changed = changed_idx.numel()
        pct_changed = (num_changed / num_total_gaussians) * 100.0

        # 2. Independent Reference Full Rerender
        colors_ref, _, _ = gsplat.rasterization(
            perturbed_scene["means"], perturbed_scene["quats"], perturbed_scene["scales"],
            perturbed_scene["opacities"], perturbed_scene["colors"],
            perturbed_scene["viewmats"], perturbed_scene["Ks"], width, height
        )

        # 3. Ground-Truth Changed Tiles (Vectorized Max-Pool)
        gt_tiles = compute_ground_truth_changed_tiles(cached_frame, colors_ref[0], tile_size=tile_size)
        gt_tile_count = int(gt_tiles.sum().item())

        # 4. Invalidation Mask Computation (Timed Stage)
        def fn_invalidation():
            return compute_tile_invalidation_mask(
                scene, perturbed_scene, changed_idx, tile_size=tile_size, compute_stats=False
            )
        dist_inv = measure_stage_distribution(fn_invalidation, warmup_reps, timing_reps)
        invalid_mask, pixel_mask, _ = fn_invalidation()
        inv_stats = compute_stats_from_mask(invalid_mask, tile_size, width, height)

        invalid_tile_count = inv_stats["invalid_tile_count"]
        invalid_pct = inv_stats["invalid_tile_percentage"]
        clean_pct = inv_stats["clean_tile_percentage"]

        # 5. Coverage Verification (Zero False Negatives)
        cov_eval = evaluate_invalidation_coverage(invalid_mask, gt_tiles)
        assert cov_eval["zero_false_negatives"], f"CRITICAL: Missed {cov_eval['false_negatives']} ground-truth changed tiles!"

        # 6. Full Projection Stage
        def fn_proj_full():
            return gsplat.fully_fused_projection(
                perturbed_scene["means"], None, perturbed_scene["quats"], perturbed_scene["scales"],
                perturbed_scene["viewmats"], perturbed_scene["Ks"], width, height, packed=False
            )
        dist_proj_full = measure_stage_distribution(fn_proj_full, warmup_reps, timing_reps)
        radii_p, means2d_p, depths_p, conics_p, _ = fn_proj_full()

        # 7. Tile Intersection & Sorting Stage
        def fn_sort_full():
            return gsplat.isect_tiles(means2d_p, radii_p, depths_p, tile_size, tile_w, tile_h)
        dist_sort_full = measure_stage_distribution(fn_sort_full, warmup_reps, timing_reps)
        tiles_per_gauss, isect_ids, flatten_ids = fn_sort_full()

        # 8. Tile Offset Encoding Stage
        def fn_offset_encode():
            return gsplat.isect_offset_encode(isect_ids, 1, tile_w, tile_h)
        dist_offset_encode = measure_stage_distribution(fn_offset_encode, warmup_reps, timing_reps)
        isect_offsets = fn_offset_encode()

        pert_colors = perturbed_scene["colors"][None, ...]
        pert_opacities = perturbed_scene["opacities"][None, ...]

        # 9. Full Rasterization Kernel Stage
        def fn_raster_full():
            return gsplat.rasterize_to_pixels(
                means2d_p, conics_p, pert_colors, pert_opacities,
                width, height, tile_size, isect_offsets, flatten_ids
            )
        dist_raster_full = measure_stage_distribution(fn_raster_full, warmup_reps, timing_reps)

        # 10. Tile-Masked Rasterization Kernel Stage
        def fn_raster_masked():
            return gsplat.rasterize_to_pixels(
                means2d_p, conics_p, pert_colors, pert_opacities,
                width, height, tile_size, isect_offsets, flatten_ids,
                masks=invalid_mask[None, ...]
            )
        dist_raster_masked = measure_stage_distribution(fn_raster_masked, warmup_reps, timing_reps)
        sparse_colors, _ = fn_raster_masked()

        # 11. Frame Compositing Stage
        def fn_composite():
            return composite_with_cache(cached_frame, sparse_colors[0], pixel_mask)
        dist_composite = measure_stage_distribution(fn_composite, warmup_reps, timing_reps)
        composited = fn_composite()

        # 12. End-to-End Baseline Full Rerender
        def fn_total_full():
            return gsplat.rasterization(
                perturbed_scene["means"], perturbed_scene["quats"], perturbed_scene["scales"],
                perturbed_scene["opacities"], perturbed_scene["colors"],
                perturbed_scene["viewmats"], perturbed_scene["Ks"], width, height
            )
        dist_total_full = measure_stage_distribution(fn_total_full, warmup_reps, timing_reps)

        # 13. End-to-End Incremental Pipeline
        def fn_total_incremental():
            inv_m, px_m, _ = compute_tile_invalidation_mask(
                scene, perturbed_scene, changed_idx, tile_size=tile_size, compute_stats=False
            )
            rad, m2d, dep, con, _ = gsplat.fully_fused_projection(
                perturbed_scene["means"], None, perturbed_scene["quats"], perturbed_scene["scales"],
                perturbed_scene["viewmats"], perturbed_scene["Ks"], width, height, packed=False
            )
            _, isects, flats = gsplat.isect_tiles(m2d, rad, dep, tile_size, tile_w, tile_h)
            offs = gsplat.isect_offset_encode(isects, 1, tile_w, tile_h)
            cols, _ = gsplat.rasterize_to_pixels(
                m2d, con, pert_colors, pert_opacities,
                width, height, tile_size, offs, flats, masks=inv_m[None, ...]
            )
            return composite_with_cache(cached_frame, cols[0], px_m)
        dist_total_incremental = measure_stage_distribution(fn_total_incremental, warmup_reps, timing_reps)

        # 14. Image Quality Metrics
        img_metrics = compute_image_metrics(colors_ref[0], composited)

        # Speedups
        raster_kernel_speedup = dist_raster_full["median"] / dist_raster_masked["median"] if dist_raster_masked["median"] > 0 else 1.0
        e2e_speedup = dist_total_full["median"] / dist_total_incremental["median"] if dist_total_incremental["median"] > 0 else 1.0

        # Construct CSV row
        csv_row = {
            "condition_id": cond["id"],
            "condition_name": cond["name"],
            "num_total_gaussians": num_total_gaussians,
            "num_changed_gaussians": num_changed,
            "pct_changed_gaussians": round(pct_changed, 3),
            "total_tiles": total_tiles,
            "gt_changed_tiles": gt_tile_count,
            "invalid_tile_count": invalid_tile_count,
            "invalid_tile_pct": round(invalid_pct, 2),
            "clean_tile_pct": round(clean_pct, 2),
            "zero_false_negatives": cov_eval["zero_false_negatives"],
            "recall": round(cov_eval["recall"], 4),
            "precision": round(cov_eval["precision"], 4),
            "t_proj_full_ms_median": round(dist_proj_full["median"], 4),
            "t_proj_full_ms_std": round(dist_proj_full["std"], 4),
            "t_sort_full_ms_median": round(dist_sort_full["median"], 4),
            "t_sort_full_ms_std": round(dist_sort_full["std"], 4),
            "t_offset_encode_ms_median": round(dist_offset_encode["median"], 4),
            "t_offset_encode_ms_std": round(dist_offset_encode["std"], 4),
            "t_raster_full_ms_median": round(dist_raster_full["median"], 4),
            "t_raster_full_ms_std": round(dist_raster_full["std"], 4),
            "t_invalidation_ms_median": round(dist_inv["median"], 4),
            "t_invalidation_ms_std": round(dist_inv["std"], 4),
            "t_invalidation_ms_iqr": round(dist_inv["iqr"], 4),
            "t_raster_masked_ms_median": round(dist_raster_masked["median"], 4),
            "t_raster_masked_ms_std": round(dist_raster_masked["std"], 4),
            "raster_kernel_speedup": round(raster_kernel_speedup, 2),
            "t_composite_ms_median": round(dist_composite["median"], 4),
            "t_composite_ms_std": round(dist_composite["std"], 4),
            "t_total_full_render_ms_median": round(dist_total_full["median"], 4),
            "t_total_full_render_ms_std": round(dist_total_full["std"], 4),
            "t_total_incremental_pipeline_ms_median": round(dist_total_incremental["median"], 4),
            "t_total_incremental_pipeline_ms_std": round(dist_total_incremental["std"], 4),
            "end_to_end_speedup": round(e2e_speedup, 2),
            "max_abs_error": round(img_metrics["max_absolute_error"], 8),
            "psnr_db": round(img_metrics["psnr_db"], 2) if not math.isinf(img_metrics["psnr_db"]) else "inf",
            "ssim": round(img_metrics["ssim"], 6),
        }
        csv_rows.append(csv_row)

        json_item = {
            "condition": cond,
            "metrics": {
                "num_changed_gaussians": num_changed,
                "pct_changed_gaussians": pct_changed,
                "gt_changed_tiles": gt_tile_count,
                "invalid_tile_count": invalid_tile_count,
                "invalid_tile_pct": invalid_pct,
                "clean_tile_pct": clean_pct,
                "zero_false_negatives": cov_eval["zero_false_negatives"],
                "recall": cov_eval["recall"],
                "precision": cov_eval["precision"],
                "max_abs_error": img_metrics["max_absolute_error"],
                "psnr_db": img_metrics["psnr_db"],
                "ssim": img_metrics["ssim"],
            },
            "timings_ms": {
                "proj_full": dist_proj_full,
                "sort_full": dist_sort_full,
                "offset_encode": dist_offset_encode,
                "raster_full": dist_raster_full,
                "invalidation": dist_inv,
                "raster_masked": dist_raster_masked,
                "composite": dist_composite,
                "total_full_render": dist_total_full,
                "total_incremental_pipeline": dist_total_incremental,
            },
            "speedups": {
                "raster_kernel_speedup": raster_kernel_speedup,
                "end_to_end_speedup": e2e_speedup,
            },
        }
        json_results.append(json_item)

        print(f"  Changed Gaussians:      {num_changed} ({pct_changed:.2f}%)")
        print(f"  Invalidated Tiles:      {invalid_tile_count}/{total_tiles} ({invalid_pct:.1f}% dirty, {clean_pct:.1f}% clean)")
        print(f"  Zero False Negatives:   {cov_eval['zero_false_negatives']} (Recall={cov_eval['recall']*100:.1f}%)")
        print(f"  Invalidation Latency:   {dist_inv['median']:.4f} ms ± {dist_inv['std']:.4f} ms (IQR={dist_inv['iqr']:.4f} ms)")
        print(f"  Raster Kernel Full:     {dist_raster_full['median']:.4f} ms ± {dist_raster_full['std']:.4f} ms")
        print(f"  Raster Kernel Masked:   {dist_raster_masked['median']:.4f} ms ± {dist_raster_masked['std']:.4f} ms  [Kernel Speedup: {raster_kernel_speedup:.2f}x]")
        print(f"  Compositing Latency:    {dist_composite['median']:.4f} ms ± {dist_composite['std']:.4f} ms")
        print(f"  Total Full Rerender:    {dist_total_full['median']:.4f} ms ± {dist_total_full['std']:.4f} ms")
        print(f"  Total Incremental Pass: {dist_total_incremental['median']:.4f} ms ± {dist_total_incremental['std']:.4f} ms  [E2E Ratio: {e2e_speedup:.2f}x]")
        print(f"  Fidelity vs Reference:  PSNR={csv_row['psnr_db']} dB | SSIM={img_metrics['ssim']:.6f} | Max Error={img_metrics['max_absolute_error']:.6f}")

        # Save visual comparison for the 1% cluster condition
        if cond["id"] == "exp2_cluster_1pct":
            _save_visualizations(
                output_dir, cached_frame, colors_ref[0], composited,
                invalid_mask, gt_tiles, tile_size
            )

    # Save CSV and JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "invalidation_experiment.csv"
    json_path = output_dir / "invalidation_experiment.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "device": gpu_name,
                "pytorch_version": torch.__version__,
                "gsplat_version": gsplat.__version__,
                "timing_repetitions": timing_reps,
                "warmup_repetitions": warmup_reps,
                "resolution": f"{width}x{height}",
                "tile_size": f"{tile_size}x{tile_size}",
                "total_tiles": total_tiles,
                "results": json_results,
            },
            f,
            indent=2,
        )

    print("\n=========================================================================================")
    print("EXPERIMENT EXECUTION COMPLETE")
    print("=========================================================================================")
    print(f"Saved CSV metrics:     {csv_path}")
    print(f"Saved JSON metrics:    {json_path}")
    print("=========================================================================================")


def _save_visualizations(output_dir, img_old, img_ref, img_comp, invalid_mask, gt_tiles, tile_size):
    """Generates visual difference and invalidation tile overlay images."""
    height, width, _ = img_old.shape

    # 1. Pixel difference map (magnified 5x for visibility)
    diff = (img_ref - img_old).abs().max(dim=-1).values
    diff_vis = (diff.clamp(0.0, 1.0) * 5.0).clamp(0.0, 1.0)
    diff_vis_np = (diff_vis.cpu().numpy() * 255.0).astype(np.uint8)

    # 2. Predicted invalid tiles overlay (cyan overlay on dimmed baseline)
    pred_pixel = invalid_mask.repeat_interleave(tile_size, dim=0).repeat_interleave(tile_size, dim=1)[:height, :width]
    base_vis = (img_old.cpu().numpy() * 255.0).astype(np.uint8)
    overlay = base_vis.copy()
    overlay[pred_pixel.cpu().numpy()] = (
        overlay[pred_pixel.cpu().numpy()] * 0.4 + np.array([0, 200, 255]) * 0.6
    ).astype(np.uint8)

    # 3. Ground-truth changed tiles overlay (magenta)
    gt_pixel = gt_tiles.repeat_interleave(tile_size, dim=0).repeat_interleave(tile_size, dim=1)[:height, :width]
    gt_overlay = base_vis.copy()
    gt_overlay[gt_pixel.cpu().numpy()] = (
        gt_overlay[gt_pixel.cpu().numpy()] * 0.4 + np.array([255, 50, 150]) * 0.6
    ).astype(np.uint8)

    # 4. Save individual renders
    comp_np = (img_comp.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    ref_np = (img_ref.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)

    Image.fromarray(diff_vis_np, mode="L").save(str(output_dir / "invalidation_exp_diff_map.png"))
    Image.fromarray(overlay, mode="RGB").save(str(output_dir / "invalidation_exp_tile_mask.png"))
    Image.fromarray(gt_overlay, mode="RGB").save(str(output_dir / "invalidation_exp_gt_tiles.png"))
    Image.fromarray(comp_np, mode="RGB").save(str(output_dir / "invalidation_exp_composite.png"))

    # 5. Build side-by-side 2x2 comparison grid
    grid = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
    grid[:height, :width] = base_vis
    grid[:height, width:] = ref_np
    grid[height:, :width] = overlay
    grid[height:, width:] = comp_np

    Image.fromarray(grid, mode="RGB").save(str(output_dir / "invalidation_exp_summary_grid.png"))
    print(f"  [Artifacts] Saved comparison images to {output_dir}")


if __name__ == "__main__":
    run_benchmark(
        scene_path=ROOT / "results" / "procedural_scene_5k.pt",
        output_dir=ROOT / "results",
        warmup_reps=15,
        timing_reps=50,
    )
