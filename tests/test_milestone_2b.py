"""
test_milestone_2b.py - Milestone 2B Static/Dynamic Partitioning & Cache Parity Tests

Validates:
1. StaticSceneCache initialization and validity checks (camera motion, resolution, intrinsics).
2. Dynamic subset projection (zero re-projection of static Gaussians).
3. In-place working buffer updates and resets.
4. Single-Gaussian modification parity (PSNR = inf, SSIM = 1.0, 0 false negatives).
5. Clustered modification parity (1% spatial cluster).
6. Random subset modifications (2% and 5% distributed subsets).
7. Screen-boundary and off-screen robustness.
"""

import sys
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def get_scene():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA required for Milestone 2B tests"
    scene_path = ROOT / "results" / "procedural_scene_5k.pt"
    scene = torch.load(str(scene_path), weights_only=False)
    for k, v in scene.items():
        if isinstance(v, torch.Tensor):
            scene[k] = v.to(device)
    return scene


def test_cache_initialization_and_validity():
    scene = get_scene()
    width, height = int(scene["width"]), int(scene["height"])
    cache = StaticSceneCache(scene, tile_size=16)

    # 1. Identical camera & resolution must be valid
    assert cache.is_valid(scene["viewmats"], scene["Ks"], width, height, 16), "Cache must be valid for matching camera."

    # 2. Camera pose translation must invalidate cache
    moved_viewmat = scene["viewmats"].clone()
    moved_viewmat[0, 0, 3] += 0.5
    assert not cache.is_valid(moved_viewmat, scene["Ks"], width, height, 16), "Camera translation must invalidate cache."

    # 3. Camera intrinsics change must invalidate cache
    altered_Ks = scene["Ks"].clone()
    altered_Ks[0, 0, 0] *= 1.1
    assert not cache.is_valid(scene["viewmats"], altered_Ks, width, height, 16), "Intrinsics change must invalidate cache."

    # 4. Resolution change must invalidate cache
    assert not cache.is_valid(scene["viewmats"], scene["Ks"], 256, 256, 16), "Resolution change must invalidate cache."

    # 5. Tile size change must invalidate cache
    assert not cache.is_valid(scene["viewmats"], scene["Ks"], width, height, 32), "Tile size change must invalidate cache."

    print("  [PASS] test_cache_initialization_and_validity")


def test_single_gaussian_partitioning():
    scene = get_scene()
    width, height = int(scene["width"]), int(scene["height"])
    tile_size = 16
    tile_w = width // tile_size
    tile_h = height // tile_size

    cache = StaticSceneCache(scene, tile_size=tile_size)

    # Perturb 1 single Gaussian
    perturbed, changed_idx, _ = perturb_scene(
        scene, fraction=0.0, delta=(0.10, -0.08, 0.02), seed=42, mode="single"
    )
    assert changed_idx.numel() == 1, f"Expected 1 changed Gaussian, got {changed_idx.numel()}"

    # Independent ground-truth full rerender
    img_ref, _, _ = gsplat.rasterization(
        perturbed["means"], perturbed["quats"], perturbed["scales"],
        perturbed["opacities"], perturbed["colors"],
        perturbed["viewmats"], perturbed["Ks"], width, height
    )

    # Dynamic projection: only 1 Gaussian projected
    rad_dyn, m2d_dyn, dep_dyn, con_dyn = cache.project_dynamic_subset(perturbed, changed_idx)
    assert rad_dyn.shape == (1, 1, 2)
    assert m2d_dyn.shape == (1, 1, 2)
    assert dep_dyn.shape == (1, 1)
    assert con_dyn.shape == (1, 1, 3)

    # Invalidation directly from cache
    inv_mask, px_mask = cache.compute_invalidation_from_cache(changed_idx, rad_dyn, m2d_dyn, dep_dyn)

    # Verify zero false negatives against pixel difference
    gt_tiles = compute_ground_truth_changed_tiles(cache.cached_frame, img_ref[0], tile_size=tile_size)
    eval_res = evaluate_invalidation_coverage(inv_mask, gt_tiles)
    assert eval_res["zero_false_negatives"], f"Missed {eval_res['false_negatives']} changed tiles!"

    # Working buffer in-place update & selective render
    cache.update_working_buffers(changed_idx, rad_dyn, m2d_dyn, dep_dyn, con_dyn)
    _, isects, flats = gsplat.isect_tiles(cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h)
    offs = gsplat.isect_offset_encode(isects, 1, tile_w, tile_h)
    cols, _ = gsplat.rasterize_to_pixels(
        cache.buf_means2d, cache.buf_conics, perturbed["colors"][None, ...], perturbed["opacities"][None, ...],
        width, height, tile_size, offs, flats, masks=inv_mask[None, ...]
    )
    comp = composite_with_cache(cache.cached_frame, cols[0], px_mask)

    # Parity check
    metrics = compute_image_metrics(img_ref[0], comp)
    assert metrics["max_absolute_error"] == 0.0, f"Max abs error: {metrics['max_absolute_error']}"
    assert math.isinf(metrics["psnr_db"]), f"PSNR: {metrics['psnr_db']}"
    assert metrics["ssim"] == 1.0, f"SSIM: {metrics['ssim']}"

    # Reset buffer and verify restoration
    cache.reset_working_buffers_to_cache(changed_idx)
    assert torch.equal(cache.buf_means2d, cache.cached_means2d)
    assert torch.equal(cache.buf_radii, cache.cached_radii)

    print("  [PASS] test_single_gaussian_partitioning")


def test_spatial_cluster_partitioning():
    scene = get_scene()
    width, height = int(scene["width"]), int(scene["height"])
    tile_size = 16
    tile_w = width // tile_size
    tile_h = height // tile_size

    cache = StaticSceneCache(scene, tile_size=tile_size)

    # 1% spatial cluster (50 Gaussians)
    perturbed, changed_idx, _ = perturb_scene(
        scene, fraction=0.01, delta=(0.15, -0.10, 0.05), seed=42, mode="spatial_cluster"
    )
    assert changed_idx.numel() == 50

    img_ref, _, _ = gsplat.rasterization(
        perturbed["means"], perturbed["quats"], perturbed["scales"],
        perturbed["opacities"], perturbed["colors"],
        perturbed["viewmats"], perturbed["Ks"], width, height
    )

    rad_dyn, m2d_dyn, dep_dyn, con_dyn = cache.project_dynamic_subset(perturbed, changed_idx)
    inv_mask, px_mask = cache.compute_invalidation_from_cache(changed_idx, rad_dyn, m2d_dyn, dep_dyn)

    gt_tiles = compute_ground_truth_changed_tiles(cache.cached_frame, img_ref[0], tile_size=tile_size)
    eval_res = evaluate_invalidation_coverage(inv_mask, gt_tiles)
    assert eval_res["zero_false_negatives"], f"Missed {eval_res['false_negatives']} changed tiles!"

    cache.update_working_buffers(changed_idx, rad_dyn, m2d_dyn, dep_dyn, con_dyn)
    _, isects, flats = gsplat.isect_tiles(cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h)
    offs = gsplat.isect_offset_encode(isects, 1, tile_w, tile_h)
    cols, _ = gsplat.rasterize_to_pixels(
        cache.buf_means2d, cache.buf_conics, perturbed["colors"][None, ...], perturbed["opacities"][None, ...],
        width, height, tile_size, offs, flats, masks=inv_mask[None, ...]
    )
    comp = composite_with_cache(cache.cached_frame, cols[0], px_mask)

    metrics = compute_image_metrics(img_ref[0], comp)
    assert metrics["max_absolute_error"] == 0.0, f"Max abs error: {metrics['max_absolute_error']}"
    assert math.isinf(metrics["psnr_db"]), f"PSNR: {metrics['psnr_db']}"
    assert metrics["ssim"] == 1.0, f"SSIM: {metrics['ssim']}"

    cache.reset_working_buffers_to_cache(changed_idx)
    print("  [PASS] test_spatial_cluster_partitioning")


def test_random_subset_partitioning():
    scene = get_scene()
    width, height = int(scene["width"]), int(scene["height"])
    tile_size = 16
    tile_w = width // tile_size
    tile_h = height // tile_size

    cache = StaticSceneCache(scene, tile_size=tile_size)

    for frac in [0.02, 0.05]:
        perturbed, changed_idx, _ = perturb_scene(
            scene, fraction=frac, delta=(0.10, 0.10, -0.05), seed=123, mode="random_subset"
        )
        img_ref, _, _ = gsplat.rasterization(
            perturbed["means"], perturbed["quats"], perturbed["scales"],
            perturbed["opacities"], perturbed["colors"],
            perturbed["viewmats"], perturbed["Ks"], width, height
        )

        rad_dyn, m2d_dyn, dep_dyn, con_dyn = cache.project_dynamic_subset(perturbed, changed_idx)
        inv_mask, px_mask = cache.compute_invalidation_from_cache(changed_idx, rad_dyn, m2d_dyn, dep_dyn)

        gt_tiles = compute_ground_truth_changed_tiles(cache.cached_frame, img_ref[0], tile_size=tile_size)
        eval_res = evaluate_invalidation_coverage(inv_mask, gt_tiles)
        assert eval_res["zero_false_negatives"]

        cache.update_working_buffers(changed_idx, rad_dyn, m2d_dyn, dep_dyn, con_dyn)
        _, isects, flats = gsplat.isect_tiles(cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h)
        offs = gsplat.isect_offset_encode(isects, 1, tile_w, tile_h)
        cols, _ = gsplat.rasterize_to_pixels(
            cache.buf_means2d, cache.buf_conics, perturbed["colors"][None, ...], perturbed["opacities"][None, ...],
            width, height, tile_size, offs, flats, masks=inv_mask[None, ...]
        )
        comp = composite_with_cache(cache.cached_frame, cols[0], px_mask)

        metrics = compute_image_metrics(img_ref[0], comp)
        assert metrics["max_absolute_error"] == 0.0
        assert math.isinf(metrics["psnr_db"])
        assert metrics["ssim"] == 1.0

        cache.reset_working_buffers_to_cache(changed_idx)
        print(f"  [PASS] test_random_subset_partitioning ({frac*100:.0f}% changed: {changed_idx.numel()} Gaussians)")


def test_boundary_crossing_robustness():
    scene = get_scene()
    width, height = int(scene["width"]), int(scene["height"])
    tile_size = 16
    tile_w = width // tile_size
    tile_h = height // tile_size

    cache = StaticSceneCache(scene, tile_size=tile_size)

    # Large delta pushing Gaussians off-screen
    perturbed, changed_idx, _ = perturb_scene(
        scene, fraction=0.02, delta=(3.0, -2.5, 0.5), seed=77, mode="random_subset"
    )
    img_ref, _, _ = gsplat.rasterization(
        perturbed["means"], perturbed["quats"], perturbed["scales"],
        perturbed["opacities"], perturbed["colors"],
        perturbed["viewmats"], perturbed["Ks"], width, height
    )

    rad_dyn, m2d_dyn, dep_dyn, con_dyn = cache.project_dynamic_subset(perturbed, changed_idx)
    inv_mask, px_mask = cache.compute_invalidation_from_cache(changed_idx, rad_dyn, m2d_dyn, dep_dyn)

    gt_tiles = compute_ground_truth_changed_tiles(cache.cached_frame, img_ref[0], tile_size=tile_size)
    eval_res = evaluate_invalidation_coverage(inv_mask, gt_tiles)
    assert eval_res["zero_false_negatives"]

    cache.update_working_buffers(changed_idx, rad_dyn, m2d_dyn, dep_dyn, con_dyn)
    _, isects, flats = gsplat.isect_tiles(cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h)
    offs = gsplat.isect_offset_encode(isects, 1, tile_w, tile_h)
    cols, _ = gsplat.rasterize_to_pixels(
        cache.buf_means2d, cache.buf_conics, perturbed["colors"][None, ...], perturbed["opacities"][None, ...],
        width, height, tile_size, offs, flats, masks=inv_mask[None, ...]
    )
    comp = composite_with_cache(cache.cached_frame, cols[0], px_mask)

    metrics = compute_image_metrics(img_ref[0], comp)
    assert metrics["max_absolute_error"] == 0.0
    assert math.isinf(metrics["psnr_db"])
    assert metrics["ssim"] == 1.0

    cache.reset_working_buffers_to_cache(changed_idx)
    print("  [PASS] test_boundary_crossing_robustness")


if __name__ == "__main__":
    print("Running Milestone 2B Static/Dynamic Partitioning Test Suite...")
    test_cache_initialization_and_validity()
    test_single_gaussian_partitioning()
    test_spatial_cluster_partitioning()
    test_random_subset_partitioning()
    test_boundary_crossing_robustness()
    print("\nALL Milestone 2B TESTS PASSED WITH 100% MATHEMATICAL PARITY!")
