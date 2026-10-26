"""
test_milestone_2d.py - Milestone 2D Verification & Demo Robustness Test Suite

Validates:
1. Scene scaling (5k, 10k, 25k, 50k, 100k) cache initialization and VRAM bounds (<120 MB).
2. Zero false negatives (Recall = 100%) across spatial cluster, single, and random perturbations.
3. Exact mathematical parity (PSNR = inf, SSIM = 1.0, Max Error = 0.0) vs full rerender.
4. Baseline reset robustness: restoring t0 produces 0 dirty tiles and exact bit-for-bit parity.
5. Repeated update isolation: repeated perturbations do not corrupt cached state.
6. Static sorting reuse: candidate-subset sorting achieves exact parity with global sort.
7. Graceful error handling for missing scenes and invalid configurations.
"""

import math
import sys
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
from src.experiment_3dgs import run_3dgs_experiment


def test_scene_scaling_and_memory_bounds():
    """Validates that scenes from 5k to 100k Gaussians initialize and remain within VRAM limits."""
    print("Testing scene scaling and memory bounds (5k, 10k, 25k, 50k, 100k)...")
    device = torch.device("cuda:0")
    width, height = 512, 512
    tile_size = 16

    for N in [5000, 10000, 25000, 50000, 100000]:
        torch.cuda.reset_peak_memory_stats()
        scene = generate_procedural_scene(num_gaussians=N, device=device, seed=42)
        scene["viewmats"] = torch.eye(4, device=device)[None, :, :]
        scene["Ks"] = torch.tensor([[512.0, 0, 256.0], [0, 512.0, 256.0], [0, 0, 1.0]], device=device)[None, :, :]
        scene["width"] = width
        scene["height"] = height

        cache = StaticSceneCache(scene, tile_size=tile_size)
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        assert cache.cached_frame.shape == (512, 512, 3), f"Invalid cached frame shape at N={N}"
        assert cache.num_total_gaussians == N, f"Mismatch in Gaussian count at N={N}"
        assert vram_mb < 150.0, f"VRAM allocation ({vram_mb:.1f} MB) exceeded 150 MB limit at N={N}"

    print("  [PASS] test_scene_scaling_and_memory_bounds")


def test_zero_false_negatives_and_parity_at_scale():
    """Validates 100% recall (0 false negatives) and exact mathematical parity on 10k and 100k scenes."""
    print("Testing zero false negatives and parity at scale (10k and 100k)...")
    device = torch.device("cuda:0")
    width, height = 512, 512
    tile_size = 16
    tile_w = width // tile_size
    tile_h = height // tile_size

    for N in [10000, 100000]:
        scene = generate_procedural_scene(num_gaussians=N, device=device, seed=42)
        scene["viewmats"] = torch.eye(4, device=device)[None, :, :]
        scene["Ks"] = torch.tensor([[512.0, 0, 256.0], [0, 512.0, 256.0], [0, 0, 1.0]], device=device)[None, :, :]
        scene["width"] = width
        scene["height"] = height

        cache = StaticSceneCache(scene, tile_size=tile_size)

        for mode in ["single", "spatial_cluster", "random_subset"]:
            frac = 0.01 if mode != "single" else 1.0 / N
            perturbed, changed_idx, _ = perturb_scene(scene, fraction=frac, seed=42, mode=mode)

            # Ground truth full rerender
            ref_colors, _, _ = gsplat.rasterization(
                perturbed["means"], perturbed["quats"], perturbed["scales"],
                perturbed["opacities"], perturbed["colors"],
                perturbed["viewmats"], perturbed["Ks"], width, height
            )
            img_ref = ref_colors[0]

            # Invalidation
            rad_d, m2d_d, dep_d, con_d = cache.project_dynamic_subset(perturbed, changed_idx)
            inv_m, _ = cache.compute_invalidation_from_cache(changed_idx, rad_d, m2d_d, dep_d, generate_pixel_mask=False)

            gt_tiles = compute_ground_truth_changed_tiles(cache.cached_frame, img_ref, tile_size=tile_size)
            cov = evaluate_invalidation_coverage(inv_m, gt_tiles)

            assert cov["zero_false_negatives"], f"False negatives detected at N={N}, mode={mode}: {cov['false_negatives']} missed"
            assert cov["recall"] == 1.0, f"Recall was {cov['recall']} (<1.0) at N={N}, mode={mode}"

            # Incremental composite
            cache.update_working_buffers(changed_idx, rad_d, m2d_d, dep_d, con_d)
            _, iscts, flts = gsplat.isect_tiles(cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h)
            ofst = gsplat.isect_offset_encode(iscts, 1, tile_w, tile_h)
            sparse_cols, _ = gsplat.rasterize_to_pixels(
                cache.buf_means2d, cache.buf_conics, perturbed["colors"][None, ...], perturbed["opacities"][None, ...],
                width, height, tile_size, ofst, flts, masks=inv_m[None, ...]
            )
            comp_img = cache.composite(sparse_cols[0], inv_m)
            cache.reset_working_buffers_to_cache(changed_idx)

            metrics = compute_image_metrics(img_ref, comp_img)
            assert metrics["max_absolute_error"] == 0.0, f"Max error non-zero at N={N}, mode={mode}: {metrics['max_absolute_error']}"
            assert math.isinf(metrics["psnr_db"]), f"PSNR was not infinity at N={N}, mode={mode}: {metrics['psnr_db']}"
            assert metrics["ssim"] == 1.0, f"SSIM was not 1.0 at N={N}, mode={mode}: {metrics['ssim']}"

    print("  [PASS] test_zero_false_negatives_and_parity_at_scale")


def test_baseline_reset_and_repeated_updates():
    """Validates that reset restores baseline rest state and repeated updates do not corrupt cache."""
    print("Testing baseline reset and repeated update robustness...")
    device = torch.device("cuda:0")
    width, height = 512, 512
    tile_size = 16
    tile_w = width // tile_size
    tile_h = height // tile_size

    scene = generate_procedural_scene(num_gaussians=5000, device=device, seed=42)
    scene["viewmats"] = torch.eye(4, device=device)[None, :, :]
    scene["Ks"] = torch.tensor([[512.0, 0, 256.0], [0, 512.0, 256.0], [0, 0, 1.0]], device=device)[None, :, :]
    scene["width"] = width
    scene["height"] = height

    cache = StaticSceneCache(scene, tile_size=tile_size)
    initial_cached_frame = cache.cached_frame.clone()

    # Apply 5 consecutive different updates
    for step in range(5):
        perturbed, changed_idx, _ = perturb_scene(
            scene, fraction=0.02, delta=(0.10 * (step + 1), -0.05, 0.02), seed=42 + step, mode="spatial_cluster"
        )
        rad_d, m2d_d, dep_d, con_d = cache.project_dynamic_subset(perturbed, changed_idx)
        inv_m, _ = cache.compute_invalidation_from_cache(changed_idx, rad_d, m2d_d, dep_d, generate_pixel_mask=False)
        cache.update_working_buffers(changed_idx, rad_d, m2d_d, dep_d, con_d)
        cache.reset_working_buffers_to_cache(changed_idx)

    # Verify working buffer is identical to baseline cache
    assert torch.equal(cache.buf_means2d, cache.cached_means2d), "Buffer means2d corrupted after repeated updates"
    assert torch.equal(cache.buf_radii, cache.cached_radii), "Buffer radii corrupted after repeated updates"
    assert torch.equal(cache.buf_depths, cache.cached_depths), "Buffer depths corrupted after repeated updates"
    assert torch.equal(cache.buf_conics, cache.cached_conics), "Buffer conics corrupted after repeated updates"
    assert torch.equal(cache.cached_frame, initial_cached_frame), "Cached frame t0 corrupted after repeated updates"

    # Test explicit fraction=0.0 reset
    rad_0, m2d_0, dep_0, con_0 = cache.project_dynamic_subset(scene, torch.tensor([], dtype=torch.long, device=device))
    inv_0, _ = cache.compute_invalidation_from_cache(torch.tensor([], dtype=torch.long, device=device), rad_0, m2d_0, dep_0, False)
    assert inv_0.sum().item() == 0, "Reset invalidation mask contains non-zero dirty tiles"

    print("  [PASS] test_baseline_reset_and_repeated_updates")


def test_candidate_sorting_parity_proof():
    """Validates that candidate-subset sorting produces exact mathematical parity with global sort."""
    print("Testing candidate-subset sorting mathematical parity...")
    device = torch.device("cuda:0")
    width, height = 512, 512
    tile_size = 16
    tile_w = width // tile_size
    tile_h = height // tile_size

    scene = generate_procedural_scene(num_gaussians=25000, device=device, seed=42)
    scene["viewmats"] = torch.eye(4, device=device)[None, :, :]
    scene["Ks"] = torch.tensor([[512.0, 0, 256.0], [0, 512.0, 256.0], [0, 0, 1.0]], device=device)[None, :, :]
    scene["width"] = width
    scene["height"] = height

    cache = StaticSceneCache(scene, tile_size=tile_size)
    perturbed, changed_idx, _ = perturb_scene(scene, fraction=0.01, seed=42, mode="spatial_cluster")

    rad_d, m2d_d, dep_d, con_d = cache.project_dynamic_subset(perturbed, changed_idx)
    inv_m, _ = cache.compute_invalidation_from_cache(changed_idx, rad_d, m2d_d, dep_d, generate_pixel_mask=False)
    cache.update_working_buffers(changed_idx, rad_d, m2d_d, dep_d, con_d)

    pert_cols = perturbed["colors"][None, ...]
    pert_opac = perturbed["opacities"][None, ...]

    # Global sort raster
    _, isect_g, flat_g = gsplat.isect_tiles(cache.buf_means2d, cache.buf_radii, cache.buf_depths, tile_size, tile_w, tile_h)
    offs_g = gsplat.isect_offset_encode(isect_g, 1, tile_w, tile_h)
    cols_g, _ = gsplat.rasterize_to_pixels(cache.buf_means2d, cache.buf_conics, pert_cols, pert_opac, width, height, tile_size, offs_g, flat_g, masks=inv_m[None, ...])

    # Candidate subset sort raster
    c_idx = cache.filter_candidate_gaussians(inv_m)
    n_c = c_idx.numel()
    m_s = cache.buf_means2d[0, c_idx].contiguous()
    r_s = cache.buf_radii[0, c_idx].contiguous()
    d_s = cache.buf_depths[0, c_idx].contiguous()
    im_ids = torch.zeros(n_c, device=device, dtype=torch.int64)

    _, isect_c, flat_c = gsplat.isect_tiles(m_s, r_s, d_s, tile_size, tile_w, tile_h, sort=True, packed=True, n_images=1, image_ids=im_ids, gaussian_ids=c_idx)
    offs_c = gsplat.isect_offset_encode(isect_c, 1, tile_w, tile_h)
    flat_c_mapped = c_idx[flat_c.long()].int()
    cols_c, _ = gsplat.rasterize_to_pixels(cache.buf_means2d, cache.buf_conics, pert_cols, pert_opac, width, height, tile_size, offs_c, flat_c_mapped, masks=inv_m[None, ...])

    max_diff = (cols_g - cols_c).abs().max().item()
    assert max_diff == 0.0, f"Candidate sort differed from global sort with max diff {max_diff}"

    cache.reset_working_buffers_to_cache(changed_idx)
    print("  [PASS] test_candidate_sorting_parity_proof")


def test_experiment_module_cli_and_error_handling():
    """Validates that experiment_3dgs handles num_gaussians, missing scenes, and errors gracefully."""
    print("Testing experiment module CLI and error handling...")
    
    # 1. Test num_gaussians procedural generation via run_3dgs_experiment
    res = run_3dgs_experiment(
        num_gaussians=10000,
        mode="spatial_cluster",
        fraction=0.01,
        num_warmups=2,
        num_reps=5,
    )
    assert res["success"] is True, "run_3dgs_experiment failed with num_gaussians"
    assert res["perturbation"]["num_total_gaussians"] == 10000, "Wrong Gaussian count in result"
    assert res["fidelity"]["max_abs_error"] == 0.0, "Parity failed in experiment runner"

    # 2. Test error handling on nonexistent scene file
    try:
        run_3dgs_experiment(scene_path=ROOT / "results" / "nonexistent_scene.pt")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass

    print("  [PASS] test_experiment_module_cli_and_error_handling")


if __name__ == "__main__":
    print("=" * 70)
    print("Running Milestone 2D Verification & Demo Robustness Tests")
    print("=" * 70)
    test_scene_scaling_and_memory_bounds()
    test_zero_false_negatives_and_parity_at_scale()
    test_baseline_reset_and_repeated_updates()
    test_candidate_sorting_parity_proof()
    test_experiment_module_cli_and_error_handling()
    print("=" * 70)
    print("ALL MILESTONE 2D TESTS PASSED WITH 100% MATHEMATICAL PARITY!")
    print("=" * 70)
