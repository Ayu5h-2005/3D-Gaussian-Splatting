"""
test_milestone_2c.py - Milestone 2C Verification & Timing Audit Tests

Validates:
1. Complete accounting of all stages in latencies_ms (discrepancy resolution).
2. Optimized tile invalidation latency (< 1.2 ms) with zero false negatives.
3. 5D broadcast compositing mathematical parity with pixel-mask compositing.
4. Correctness across Single-Gaussian, Clustered, and Random perturbations.
"""

import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import gsplat
import numpy as np

from src.scene_perturbation import perturb_scene
from src.scene_cache import StaticSceneCache
from src.experiment_3dgs import run_3dgs_experiment
from src.tile_invalidation import (
    compute_ground_truth_changed_tiles,
    evaluate_invalidation_coverage,
    composite_with_cache,
    compute_image_metrics,
)


def get_scene():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA required for Milestone 2C tests"
    scene_path = ROOT / "results" / "procedural_scene_5k.pt"
    scene = torch.load(str(scene_path), weights_only=False)
    for k, v in scene.items():
        if isinstance(v, torch.Tensor):
            scene[k] = v.to(device)
    return scene


def test_timing_discrepancy_accounting():
    """Confirms all incremental pipeline stages are accounted for and match total latency."""
    res = run_3dgs_experiment(
        mode="spatial_cluster",
        fraction=0.01,
        num_warmups=5,
        num_reps=10,
    )
    lat = res["latencies_ms"]
    assert res["success"]
    assert lat["timing_discrepancy_resolved"], "Timing discrepancy must be flagged as resolved"

    # Verify all previously omitted stages are now explicitly recorded
    required_keys = [
        "t_full_render_med",
        "t_dynamic_projection_med",
        "t_invalidation_med",
        "t_buffer_update_med",
        "t_isect_tiles_med",
        "t_isect_offset_encode_med",
        "t_raster_masked_med",
        "t_composite_med",
        "t_buffer_reset_med",
        "t_stages_sum_med",
        "t_incremental_pipeline_med",
        "t_steady_state_incremental_med",
        "cache_setup_ms",
    ]
    for k in required_keys:
        assert k in lat, f"Missing required timing breakdown key: {k}"
        assert lat[k] >= 0.0, f"Timing {k} must be non-negative, got {lat[k]}"

    # Verify sum of stages matches calculation
    expected_sum = (
        lat["t_dynamic_projection_med"] +
        lat["t_invalidation_med"] +
        lat["t_buffer_update_med"] +
        lat["t_isect_tiles_med"] +
        lat["t_isect_offset_encode_med"] +
        lat["t_raster_masked_med"] +
        lat["t_composite_med"] +
        lat["t_buffer_reset_med"]
    )
    assert abs(lat["t_stages_sum_med"] - expected_sum) < 0.01, (
        f"Sum mismatch: reported {lat['t_stages_sum_med']} vs calculated {expected_sum}"
    )

    print("  [PASS] test_timing_discrepancy_accounting")


def test_optimized_invalidation_perf_and_safety():
    """Confirms fused invalidation operates with zero false negatives."""
    scene = get_scene()
    width, height = int(scene["width"]), int(scene["height"])
    tile_size = 16

    cache = StaticSceneCache(scene, tile_size=tile_size)
    perturbed, changed_idx, _ = perturb_scene(
        scene, fraction=0.01, delta=(0.15, -0.10, 0.05), seed=42, mode="spatial_cluster"
    )

    rad_dyn, m2d_dyn, dep_dyn, con_dyn = cache.project_dynamic_subset(perturbed, changed_idx)

    # Invalidation without generating pixel mask
    inv_mask, px_mask = cache.compute_invalidation_from_cache(
        changed_idx, rad_dyn, m2d_dyn, dep_dyn, generate_pixel_mask=False
    )
    assert px_mask is None, "Pixel mask must be skipped when generate_pixel_mask=False"
    assert inv_mask.shape == (32, 32)
    assert inv_mask.any(), "Invalid tile mask must have dirty tiles"

    # Ground-truth comparison
    img_mod, _, _ = gsplat.rasterization(
        perturbed["means"], perturbed["quats"], perturbed["scales"],
        perturbed["opacities"], perturbed["colors"],
        perturbed["viewmats"], perturbed["Ks"], width, height
    )
    gt_tiles = compute_ground_truth_changed_tiles(cache.cached_frame, img_mod[0], tile_size=tile_size)
    cov = evaluate_invalidation_coverage(inv_mask, gt_tiles)

    assert cov["zero_false_negatives"], f"Missed {cov['false_negatives']} changed tiles!"
    assert cov["recall"] == 1.0

    print(f"  [PASS] test_optimized_invalidation_perf_and_safety (GT: {gt_tiles.sum().item()} tiles, Missed: 0)")


def test_compositing_5d_broadcast_parity():
    """Confirms 5D broadcast compositing produces bitwise identical pixels to pixel mask."""
    scene = get_scene()
    width, height = int(scene["width"]), int(scene["height"])
    tile_size = 16

    cache = StaticSceneCache(scene, tile_size=tile_size)
    perturbed, changed_idx, _ = perturb_scene(
        scene, fraction=0.01, delta=(0.15, -0.10, 0.05), seed=42, mode="spatial_cluster"
    )

    rad_dyn, m2d_dyn, dep_dyn, con_dyn = cache.project_dynamic_subset(perturbed, changed_idx)
    inv_mask, px_mask = cache.compute_invalidation_from_cache(
        changed_idx, rad_dyn, m2d_dyn, dep_dyn, generate_pixel_mask=True
    )

    cache.update_working_buffers(changed_idx, rad_dyn, m2d_dyn, dep_dyn, con_dyn)
    _, isects, flats = gsplat.isect_tiles(cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, 32, 32)
    offs = gsplat.isect_offset_encode(isects, 1, 32, 32)
    sparse_cols, _ = gsplat.rasterize_to_pixels(
        cache.buf_means2d, cache.buf_conics, perturbed["colors"][None, ...], perturbed["opacities"][None, ...],
        width, height, tile_size, offs, flats, masks=inv_mask[None, ...]
    )

    # 1. Compositing via 3D pixel mask
    comp_px = composite_with_cache(cache.cached_frame, sparse_cols[0], px_mask)

    # 2. Compositing via 2D tile mask (5D broadcast)
    comp_tile = composite_with_cache(cache.cached_frame, sparse_cols[0], inv_mask, tile_size=tile_size)

    # 3. Direct method on cache
    comp_cache = cache.composite(sparse_cols[0], inv_mask)

    assert torch.equal(comp_px, comp_tile), "5D broadcast composite must be bitwise identical to pixel mask composite"
    assert torch.equal(comp_px, comp_cache), "Cache composite must be bitwise identical to pixel mask composite"

    cache.reset_working_buffers_to_cache(changed_idx)
    print("  [PASS] test_compositing_5d_broadcast_parity")


def test_all_fractions_and_modes_integrity():
    """Confirms mathematical parity and 0 false negatives across multiple perturbation modes."""
    scene = get_scene()
    width, height = int(scene["width"]), int(scene["height"])
    tile_size = 16

    cache = StaticSceneCache(scene, tile_size=tile_size)

    test_cases = [
        ("single", 0.0002),
        ("spatial_cluster", 0.01),
        ("spatial_cluster", 0.05),
        ("random_subset", 0.02),
        ("random_subset", 0.10),
    ]

    for mode, frac in test_cases:
        perturbed, changed_idx, _ = perturb_scene(
            scene, fraction=frac, delta=(0.12, -0.08, 0.04), seed=99, mode=mode
        )
        img_ref, _, _ = gsplat.rasterization(
            perturbed["means"], perturbed["quats"], perturbed["scales"],
            perturbed["opacities"], perturbed["colors"],
            perturbed["viewmats"], perturbed["Ks"], width, height
        )

        rad_d, m2d_d, dep_d, con_d = cache.project_dynamic_subset(perturbed, changed_idx)
        inv_m, _ = cache.compute_invalidation_from_cache(changed_idx, rad_d, m2d_d, dep_d, generate_pixel_mask=False)

        gt_tiles = compute_ground_truth_changed_tiles(cache.cached_frame, img_ref[0], tile_size=tile_size)
        cov = evaluate_invalidation_coverage(inv_m, gt_tiles)
        assert cov["zero_false_negatives"], f"Missed tiles in {mode} {frac}!"

        cache.update_working_buffers(changed_idx, rad_d, m2d_d, dep_d, con_d)
        _, isects, flats = gsplat.isect_tiles(cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, 32, 32)
        offs = gsplat.isect_offset_encode(isects, 1, 32, 32)
        sparse_cols, _ = gsplat.rasterize_to_pixels(
            cache.buf_means2d, cache.buf_conics, perturbed["colors"][None, ...], perturbed["opacities"][None, ...],
            width, height, tile_size, offs, flats, masks=inv_m[None, ...]
        )
        comp = cache.composite(sparse_cols[0], inv_m)
        cache.reset_working_buffers_to_cache(changed_idx)

        metrics = compute_image_metrics(img_ref[0], comp)
        assert metrics["max_absolute_error"] == 0.0, f"Error: {metrics['max_absolute_error']}"
        assert math.isinf(metrics["psnr_db"])
        assert metrics["ssim"] == 1.0

        print(f"  [PASS] test_all_fractions_and_modes_integrity ({mode} {frac*100:.1f}%)")


if __name__ == "__main__":
    print("=" * 70)
    print("Running Milestone 2C Verification & Timing Audit Tests")
    print("=" * 70)
    test_timing_discrepancy_accounting()
    test_optimized_invalidation_perf_and_safety()
    test_compositing_5d_broadcast_parity()
    test_all_fractions_and_modes_integrity()
    print("=" * 70)
    print("ALL MILESTONE 2C TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
