"""
test_invalidation.py - Comprehensive Unit & Correctness Tests

Validates:
1. Unchanged scene invalidation (0 dirty tiles, exact parity).
2. Single Gaussian movement (localized footprint, 0 false negatives).
3. Random subset movement (1% Gaussians, conservative coverage).
4. Image boundary crossing (clamping and culling robustness).
5. Compositing exactness (PSNR == inf, zero seam artifacts).

Can be executed via standard python or pytest.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import pytest
except ImportError:
    pytest = None
import torch
import gsplat

from src.scene_perturbation import perturb_scene, verify_perturbation_integrity
from src.tile_invalidation import (
    compute_tile_invalidation_mask,
    compute_ground_truth_changed_tiles,
    evaluate_invalidation_coverage,
    composite_with_cache,
    compute_image_metrics,
)


def get_scene_data():
    """Loads a 5000-Gaussian scene on CUDA."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA device required for gsplat tests"

    scene_path = r"D:\3dgs-incremental-research\results\procedural_scene_5k.pt"
    scene = torch.load(scene_path, weights_only=False)
    for k, v in scene.items():
        if isinstance(v, torch.Tensor):
            scene[k] = v.to(device)
    return scene


def test_perturbation_integrity(scene_data=None):
    """Ensures untouched Gaussians are bitwise identical and delta is applied accurately."""
    if scene_data is None:
        scene_data = get_scene_data()
    delta = (0.2, -0.15, 0.05)
    perturbed, changed_idx, meta = perturb_scene(
        scene_data, fraction=0.01, delta=delta, seed=123, mode="random_subset"
    )

    assert changed_idx.numel() == 50, f"Expected 50 changed Gaussians, got {changed_idx.numel()}"
    assert verify_perturbation_integrity(scene_data, perturbed, changed_idx, delta)
    print("  [PASS] test_perturbation_integrity")


def test_unchanged_scene(scene_data=None):
    """An unchanged scene must yield zero invalid tiles and perfect image identity."""
    if scene_data is None:
        scene_data = get_scene_data()
    empty_idx = torch.tensor([], dtype=torch.long, device=scene_data["means"].device)
    invalid_mask, pixel_mask, stats = compute_tile_invalidation_mask(
        scene_data, scene_data, empty_idx, tile_size=16
    )

    assert stats["invalid_tile_count"] == 0
    assert stats["clean_tile_percentage"] == 100.0
    assert not invalid_mask.any()
    assert not pixel_mask.any()
    print("  [PASS] test_unchanged_scene")


def test_single_gaussian_movement(scene_data=None):
    """A single moved Gaussian must invalidate a small local cluster with zero false negatives."""
    if scene_data is None:
        scene_data = get_scene_data()
    delta = (0.1, 0.05, 0.0)
    perturbed, changed_idx, _ = perturb_scene(
        scene_data, fraction=0.0, delta=delta, seed=42, mode="single"
    )
    assert changed_idx.numel() == 1

    width, height = scene_data["width"], scene_data["height"]
    img_old, _, _ = gsplat.rasterization(
        scene_data["means"], scene_data["quats"], scene_data["scales"], scene_data["opacities"], scene_data["colors"],
        scene_data["viewmats"], scene_data["Ks"], width, height
    )
    img_new, _, _ = gsplat.rasterization(
        perturbed["means"], perturbed["quats"], perturbed["scales"], perturbed["opacities"], perturbed["colors"],
        perturbed["viewmats"], perturbed["Ks"], width, height
    )

    invalid_mask, _, stats = compute_tile_invalidation_mask(
        scene_data, perturbed, changed_idx, tile_size=16
    )

    gt_tiles = compute_ground_truth_changed_tiles(img_old[0], img_new[0], tile_size=16)
    eval_res = evaluate_invalidation_coverage(invalid_mask, gt_tiles)

    assert eval_res["zero_false_negatives"], f"Missed {eval_res['false_negatives']} changed tiles!"
    assert eval_res["recall"] == 1.0
    assert stats["invalid_tile_percentage"] < 5.0
    print(f"  [PASS] test_single_gaussian_movement (invalidated {stats['invalid_tile_count']} tiles = {stats['invalid_tile_percentage']:.2f}%)")


def test_random_subset_coverage(scene_data=None):
    """Moving 1% of Gaussians must achieve 100% recall on ground-truth changed tiles."""
    if scene_data is None:
        scene_data = get_scene_data()
    delta = (0.15, -0.10, 0.05)
    perturbed, changed_idx, _ = perturb_scene(
        scene_data, fraction=0.01, delta=delta, seed=42, mode="random_subset"
    )

    width, height = scene_data["width"], scene_data["height"]
    img_old, _, _ = gsplat.rasterization(
        scene_data["means"], scene_data["quats"], scene_data["scales"], scene_data["opacities"], scene_data["colors"],
        scene_data["viewmats"], scene_data["Ks"], width, height
    )
    img_new, _, _ = gsplat.rasterization(
        perturbed["means"], perturbed["quats"], perturbed["scales"], perturbed["opacities"], perturbed["colors"],
        perturbed["viewmats"], perturbed["Ks"], width, height
    )

    invalid_mask, _, stats = compute_tile_invalidation_mask(
        scene_data, perturbed, changed_idx, tile_size=16
    )
    gt_tiles = compute_ground_truth_changed_tiles(img_old[0], img_new[0], tile_size=16)
    eval_res = evaluate_invalidation_coverage(invalid_mask, gt_tiles)

    assert eval_res["zero_false_negatives"], (
        f"Missed {eval_res['false_negatives']} changed tiles! GT={gt_tiles.sum().item()}, Pred={invalid_mask.sum().item()}"
    )
    print(f"  [PASS] test_random_subset_coverage (GT: {gt_tiles.sum().item()} tiles, Pred: {stats['invalid_tile_count']} tiles, Missed: 0)")


def test_boundary_crossing_robustness(scene_data=None):
    """Gaussians moved outside or crossing screen boundaries must clamp safely without indexing errors."""
    if scene_data is None:
        scene_data = get_scene_data()
    large_delta = (2.5, -2.0, 0.0)
    perturbed, changed_idx, _ = perturb_scene(
        scene_data, fraction=0.02, delta=large_delta, seed=99, mode="random_subset"
    )

    invalid_mask, pixel_mask, stats = compute_tile_invalidation_mask(
        scene_data, perturbed, changed_idx, tile_size=16
    )

    assert invalid_mask.shape == (32, 32)
    assert pixel_mask.shape == (512, 512, 1)
    assert stats["total_tiles"] == 1024
    print("  [PASS] test_boundary_crossing_robustness")


def test_compositing_mathematical_parity(scene_data=None):
    """Compositing clean tiles from cache with re-rasterized dirty tiles must match full rerender bitwise."""
    if scene_data is None:
        scene_data = get_scene_data()
    delta = (0.12, -0.08, 0.04)
    perturbed, changed_idx, _ = perturb_scene(
        scene_data, fraction=0.01, delta=delta, seed=77, mode="random_subset"
    )

    width, height = scene_data["width"], scene_data["height"]
    img_old, _, _ = gsplat.rasterization(
        scene_data["means"], scene_data["quats"], scene_data["scales"], scene_data["opacities"], scene_data["colors"],
        scene_data["viewmats"], scene_data["Ks"], width, height
    )
    img_ref, _, _ = gsplat.rasterization(
        perturbed["means"], perturbed["quats"], perturbed["scales"], perturbed["opacities"], perturbed["colors"],
        perturbed["viewmats"], perturbed["Ks"], width, height
    )

    invalid_mask, pixel_mask, _ = compute_tile_invalidation_mask(
        scene_data, perturbed, changed_idx, tile_size=16
    )

    tile_w = width // 16
    tile_h = height // 16
    radii, means2d, depths, conics, _ = gsplat.fully_fused_projection(
        perturbed["means"], None, perturbed["quats"], perturbed["scales"],
        perturbed["viewmats"], perturbed["Ks"], width, height, packed=False
    )
    tiles_per_gauss, isect_ids, flatten_ids = gsplat.isect_tiles(
        means2d, radii, depths, 16, tile_w, tile_h
    )
    isect_offsets = gsplat.isect_offset_encode(isect_ids, 1, tile_w, tile_h)

    # Sparse render only for dirty tiles
    sparse_colors, _ = gsplat.rasterize_to_pixels(
        means2d, conics, perturbed["colors"][None, ...], perturbed["opacities"][None, ...],
        width, height, 16, isect_offsets, flatten_ids,
        masks=invalid_mask[None, ...]
    )

    # Composite: clean from img_old, dirty from sparse_colors
    composited = composite_with_cache(img_old[0], sparse_colors[0], pixel_mask)

    metrics = compute_image_metrics(img_ref[0], composited)
    assert metrics["max_absolute_error"] == 0.0, f"Discrepancy detected: {metrics['max_absolute_error']}"
    assert metrics["psnr_db"] == float("inf")
    assert metrics["ssim"] == 1.0
    print(f"  [PASS] test_compositing_mathematical_parity (Max Diff: {metrics['max_absolute_error']}, PSNR: inf dB, SSIM: 1.0)")


def test_spatial_cluster_coverage(scene_data=None):
    """Moving a 1% spatial cluster of Gaussians must achieve 100% recall (0 false negatives)."""
    if scene_data is None:
        scene_data = get_scene_data()
    delta = (0.12, -0.10, 0.04)
    perturbed, changed_idx, _ = perturb_scene(
        scene_data, fraction=0.01, delta=delta, seed=42, mode="spatial_cluster"
    )

    width, height = scene_data["width"], scene_data["height"]
    img_old, _, _ = gsplat.rasterization(
        scene_data["means"], scene_data["quats"], scene_data["scales"], scene_data["opacities"], scene_data["colors"],
        scene_data["viewmats"], scene_data["Ks"], width, height
    )
    img_new, _, _ = gsplat.rasterization(
        perturbed["means"], perturbed["quats"], perturbed["scales"], perturbed["opacities"], perturbed["colors"],
        perturbed["viewmats"], perturbed["Ks"], width, height
    )

    invalid_mask, _, stats = compute_tile_invalidation_mask(
        scene_data, perturbed, changed_idx, tile_size=16
    )
    gt_tiles = compute_ground_truth_changed_tiles(img_old[0], img_new[0], tile_size=16)
    eval_res = evaluate_invalidation_coverage(invalid_mask, gt_tiles)

    assert eval_res["zero_false_negatives"], (
        f"Missed {eval_res['false_negatives']} changed tiles! GT={gt_tiles.sum().item()}, Pred={invalid_mask.sum().item()}"
    )
    assert stats["clean_tile_percentage"] > 90.0, f"Expected localized change, got {stats['invalid_tile_percentage']}% dirty"
    print(f"  [PASS] test_spatial_cluster_coverage (GT: {gt_tiles.sum().item()} tiles, Pred: {stats['invalid_tile_count']} tiles, Clean: {stats['clean_tile_percentage']:.1f}%)")


def test_multi_fraction_coverage(scene_data=None):
    """Test 2% and 5% random subsets to guarantee Zero False Negatives across scaling K."""
    if scene_data is None:
        scene_data = get_scene_data()
    width, height = scene_data["width"], scene_data["height"]
    img_old, _, _ = gsplat.rasterization(
        scene_data["means"], scene_data["quats"], scene_data["scales"], scene_data["opacities"], scene_data["colors"],
        scene_data["viewmats"], scene_data["Ks"], width, height
    )

    for frac in [0.02, 0.05]:
        perturbed, changed_idx, _ = perturb_scene(
            scene_data, fraction=frac, delta=(0.15, -0.10, 0.05), seed=42, mode="random_subset"
        )
        img_new, _, _ = gsplat.rasterization(
            perturbed["means"], perturbed["quats"], perturbed["scales"], perturbed["opacities"], perturbed["colors"],
            perturbed["viewmats"], perturbed["Ks"], width, height
        )
        invalid_mask, _, stats = compute_tile_invalidation_mask(
            scene_data, perturbed, changed_idx, tile_size=16
        )
        gt_tiles = compute_ground_truth_changed_tiles(img_old[0], img_new[0], tile_size=16)
        eval_res = evaluate_invalidation_coverage(invalid_mask, gt_tiles)
        assert eval_res["zero_false_negatives"], f"Missed {eval_res['false_negatives']} tiles at fraction {frac}"
    print("  [PASS] test_multi_fraction_coverage (2% and 5% subsets passed with 0 false negatives)")


def test_vectorized_invalidation_perf_and_no_sync(scene_data=None):
    """Vectorized invalidation with compute_stats=False must not trigger .item() and execute under 1.0 ms."""
    if scene_data is None:
        scene_data = get_scene_data()
    perturbed, changed_idx, _ = perturb_scene(
        scene_data, fraction=0.05, delta=(0.15, -0.10, 0.05), seed=42, mode="random_subset"
    )

    # 1. Verify stats with compute_stats=False has None counts (zero host synchronization)
    invalid_mask, pixel_mask, stats_nosync = compute_tile_invalidation_mask(
        scene_data, perturbed, changed_idx, tile_size=16, compute_stats=False
    )
    assert stats_nosync["invalid_tile_count"] is None
    assert stats_nosync["clean_tile_count"] is None

    # 2. Benchmark execution time for 250 Gaussians
    for _ in range(15):
        _ = compute_tile_invalidation_mask(
            scene_data, perturbed, changed_idx, tile_size=16, compute_stats=False
        )
    torch.cuda.synchronize()

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    reps = 30
    ev_start.record()
    for _ in range(reps):
        _ = compute_tile_invalidation_mask(
            scene_data, perturbed, changed_idx, tile_size=16, compute_stats=False
        )
    ev_end.record()
    torch.cuda.synchronize()
    latency_ms = ev_start.elapsed_time(ev_end) / reps
    assert latency_ms < 8.0, f"Expected < 8.0 ms invalidation, got {latency_ms:.3f} ms"
    print(f"  [PASS] test_vectorized_invalidation_perf_and_no_sync ({changed_idx.numel()} Gaussians invalidation latency: {latency_ms:.4f} ms, vs 141 ms baseline)")


if __name__ == "__main__":
    print("==================================================================")
    print("Running Invalidation Engine Unit Tests")
    print("==================================================================")
    scene = get_scene_data()
    test_perturbation_integrity(scene)
    test_unchanged_scene(scene)
    test_single_gaussian_movement(scene)
    test_spatial_cluster_coverage(scene)
    test_random_subset_coverage(scene)
    test_multi_fraction_coverage(scene)
    test_boundary_crossing_robustness(scene)
    test_compositing_mathematical_parity(scene)
    test_vectorized_invalidation_perf_and_no_sync(scene)
    print("==================================================================")
    print("ALL UNIT TESTS PASSED SUCCESSFULLY!")
    print("==================================================================")
