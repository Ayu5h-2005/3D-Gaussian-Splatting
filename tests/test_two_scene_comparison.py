"""
test_two_scene_comparison.py - Test Suite for Mode B: Automatic Two-Scene Change Detection

Validates:
1. PLY parsing and validation (Inria format, linear format, corrupt header rejection).
2. Identical scene comparison (100% static match, 0 dirty tiles, inf PSNR).
3. Different Gaussian counts (N_A != N_B) and permuted ordering.
4. Added and removed Gaussian detection.
5. Zero false negatives (Recall = 100%) and bit-exact mathematical parity.
6. Coordinate and scale compatibility detection.
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
from src.ply_io import save_3dgs_ply, load_3dgs_ply, PLYParseError
from src.scene_comparison import match_gaussians_mutual_knn, compare_and_render_two_scenes


def test_ply_io_and_validation():
    """Validates PLY saving, loading, attribute extraction, and malformed header rejection."""
    print("Testing PLY parser, serializer, and error handling...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scene = generate_procedural_scene(200, device=device, seed=42)

    tmp_dir = ROOT / "results" / "test_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    inria_ply = tmp_dir / "test_inria.ply"
    linear_ply = tmp_dir / "test_linear.ply"

    # Test Inria format
    save_3dgs_ply(scene, inria_ply, as_inria_format=True)
    loaded_inria = load_3dgs_ply(inria_ply, device=device)
    assert loaded_inria["means"].shape == (200, 3)
    assert (scene["means"] - loaded_inria["means"]).abs().max().item() < 1e-5
    assert (scene["colors"] - loaded_inria["colors"]).abs().max().item() < 1e-4

    # Test Linear format
    save_3dgs_ply(scene, linear_ply, as_inria_format=False)
    loaded_linear = load_3dgs_ply(linear_ply, device=device)
    assert loaded_linear["means"].shape == (200, 3)
    assert (scene["means"] - loaded_linear["means"]).abs().max().item() < 1e-5

    # Test Corrupt Header rejection
    corrupt_ply = tmp_dir / "corrupt.ply"
    with open(corrupt_ply, "wb") as f:
        f.write(b"not_a_ply_file\nend_header\n123")
    try:
        load_3dgs_ply(corrupt_ply, device=device)
        assert False, "Should have raised PLYParseError"
    except PLYParseError:
        pass

    print("  [PASS] test_ply_io_and_validation")


def test_identical_scenes_comparison():
    """Validates that comparing a scene against itself yields 100% static match and 0 dirty tiles."""
    print("Testing identical scene comparison...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scene = generate_procedural_scene(1000, device=device, seed=42)

    viewmats = torch.eye(4, device=device)[None, :, :]
    Ks = torch.tensor([[512.0, 0, 256.0], [0, 512.0, 256.0], [0, 0, 1.0]], device=device)[None, :, :]

    res = compare_and_render_two_scenes(
        scene_a=scene,
        scene_b=scene,
        viewmats=viewmats,
        Ks=Ks,
        width=512,
        height=512,
        tile_size=16,
        pos_tol=0.02,
        feat_tol=0.05,
    )

    assert res["counts"]["matched_static"] == 1000, f"Expected 1000 static, got {res['counts']['matched_static']}"
    assert res["counts"]["modified"] == 0
    assert res["counts"]["removed"] == 0
    assert res["counts"]["added"] == 0
    assert res["tiles"]["invalid_tile_count"] == 0, f"Expected 0 invalid tiles, got {res['tiles']['invalid_tile_count']}"
    assert res["tiles"]["clean_tile_percentage"] == 100.0
    assert res["fidelity"]["max_abs_error"] == 0.0

    print("  [PASS] test_identical_scenes_comparison")


def test_different_counts_and_ordering():
    """Validates comparison when N_A != N_B and order is randomly permuted."""
    print("Testing scenes with different counts and permuted order...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scene_a = generate_procedural_scene(2000, device=device, seed=42)

    # Build Scene B with 50 modified, 50 removed, 100 added
    means_a = scene_a["means"]
    scales_a = scene_a["scales"]
    quats_a = scene_a["quats"]
    opacities_a = scene_a["opacities"]
    colors_a = scene_a["colors"]

    # 50 modified
    means_b = means_a.clone()
    colors_b = colors_a.clone()
    means_b[:50] += torch.tensor([0.15, -0.10, 0.05], device=device)
    colors_b[:50] = torch.tensor([1.0, 0.1, 0.1], device=device)

    # 50 removed (keep 50 to end)
    keep_mask = torch.ones(2000, dtype=torch.bool, device=device)
    keep_mask[100:150] = False
    means_b = means_b[keep_mask]
    scales_b = scales_a[keep_mask]
    quats_b = quats_a[keep_mask]
    opacities_b = opacities_a[keep_mask]
    colors_b = colors_b[keep_mask]

    # 100 added
    new_means = torch.randn((100, 3), device=device) * 0.05 + torch.tensor([0.2, 0.2, 2.5], device=device)
    new_scales = torch.ones((100, 3), device=device) * 0.012
    new_quats = torch.zeros((100, 4), device=device)
    new_quats[:, 0] = 1.0
    new_opac = torch.ones(100, device=device) * 0.85
    new_cols = torch.tensor([[0.2, 0.9, 0.2]], device=device).repeat(100, 1)

    means_b = torch.cat([means_b, new_means], dim=0)
    scales_b = torch.cat([scales_b, new_scales], dim=0)
    quats_b = torch.cat([quats_b, new_quats], dim=0)
    opacities_b = torch.cat([opacities_b, new_opac], dim=0)
    colors_b = torch.cat([colors_b, new_cols], dim=0)

    # Permute Scene B
    perm = torch.randperm(means_b.shape[0], device=device)
    scene_b = {
        "means": means_b[perm],
        "scales": scales_b[perm],
        "quats": quats_b[perm],
        "opacities": opacities_b[perm],
        "colors": colors_b[perm],
    }

    viewmats = torch.eye(4, device=device)[None, :, :]
    Ks = torch.tensor([[512.0, 0, 256.0], [0, 512.0, 256.0], [0, 0, 1.0]], device=device)[None, :, :]

    res = compare_and_render_two_scenes(
        scene_a=scene_a,
        scene_b=scene_b,
        viewmats=viewmats,
        Ks=Ks,
        width=512,
        height=512,
        tile_size=16,
        pos_tol=0.03,
        feat_tol=0.08,
    )

    assert res["counts"]["scene_a_gaussians"] == 2000
    assert res["counts"]["scene_b_gaussians"] == 2050
    assert res["tiles"]["zero_false_negatives"] is True, f"False negatives detected: {res['tiles']['false_negatives']}"
    assert res["tiles"]["recall"] == 1.0
    assert res["fidelity"]["max_abs_error"] == 0.0, f"Max error non-zero: {res['fidelity']['max_abs_error']}"
    assert res["counts"]["matching_confidence"] > 0.85

    print("  [PASS] test_different_counts_and_ordering")


def test_coordinate_incompatibility():
    """Validates that scenes positioned far apart trigger compatibility warning."""
    print("Testing coordinate incompatibility detection...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scene_a = generate_procedural_scene(500, device=device, seed=42)
    scene_b = generate_procedural_scene(500, device=device, seed=42)
    # Shift Scene B 100 units away along X
    scene_b["means"] += torch.tensor([100.0, 0.0, 0.0], device=device)

    viewmats = torch.eye(4, device=device)[None, :, :]
    Ks = torch.tensor([[512.0, 0, 256.0], [0, 512.0, 256.0], [0, 0, 1.0]], device=device)[None, :, :]

    res = compare_and_render_two_scenes(
        scene_a=scene_a,
        scene_b=scene_b,
        viewmats=viewmats,
        Ks=Ks,
        width=512,
        height=512,
        tile_size=16,
    )
    assert res["compatibility"]["compatible_coords"] is False, "Should have detected coordinate incompatibility"
    print("  [PASS] test_coordinate_incompatibility")


def test_ground_truth_classification_accuracy():
    """
    Validates classification precision, recall, and confusion matrix against known ground-truth
    identities for the benchmark sample pair (5,000 baseline vs 5,050 updated Gaussians).
    """
    print("Testing ground-truth classification accuracy and confusion counts...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ply_a = ROOT / "results" / "ply_demo" / "baseline_scene.ply"
    ply_b = ROOT / "results" / "ply_demo" / "updated_scene.ply"

    scene_a = load_3dgs_ply(ply_a, device=device)
    scene_b = load_3dgs_ply(ply_b, device=device)

    # Reconstruct ground truth indices according to generate_sample_ply_pair.py
    seed_scene = generate_procedural_scene(5000, device=device, seed=42)
    dist_from_seed = torch.norm(seed_scene["means"] - seed_scene["means"][100:101], dim=-1)
    mod_idx_50 = set(torch.topk(dist_from_seed, 50, largest=False).indices.tolist())
    rem_idx_100 = set(range(500, 600))

    # Index 562 is in both mod_idx_50 and rem_idx_100 (modified then removed)
    gt_mod_a = mod_idx_50 - rem_idx_100  # 49 Gaussians
    gt_rem_a = rem_idx_100               # 100 Gaussians
    gt_static_a = set(range(5000)) - (gt_mod_a | gt_rem_a)  # 4851 Gaussians

    # Run Two-Stage Matching
    match_res = match_gaussians_mutual_knn(scene_a, scene_b, pos_tol=0.03, feat_tol=0.08, disp_tol=0.35)

    det_static_a = set(torch.where(match_res["static_a_mask"])[0].tolist())
    det_mod_a = set(torch.where(match_res["modified_a_mask"])[0].tolist())
    det_rem_a = set(torch.where(match_res["removed_a_mask"])[0].tolist())
    num_added = match_res["num_added"]

    # Confusion counts for Scene A
    tp_static = len(gt_static_a & det_static_a)
    fp_static = len(det_static_a - gt_static_a)
    fn_static = len(gt_static_a - det_static_a)

    tp_mod = len(gt_mod_a & det_mod_a)
    fp_mod = len(det_mod_a - gt_mod_a)
    fn_mod = len(gt_mod_a - det_mod_a)

    tp_rem = len(gt_rem_a & det_rem_a)
    fp_rem = len(det_rem_a - gt_rem_a)
    fn_rem = len(gt_rem_a - det_rem_a)

    prec_static = tp_static / max(tp_static + fp_static, 1)
    rec_static = tp_static / max(tp_static + fn_static, 1)

    prec_mod = tp_mod / max(tp_mod + fp_mod, 1)
    rec_mod = tp_mod / max(tp_mod + fn_mod, 1)

    prec_rem = tp_rem / max(tp_rem + fp_rem, 1)
    rec_rem = tp_rem / max(tp_rem + fn_rem, 1)

    print(f"  Classification Evaluation (Sample PLY Pair):")
    print(f"    Static:   TP={tp_static}, FP={fp_static}, FN={fn_static} | Prec={prec_static:.4f}, Rec={rec_static:.4f}")
    print(f"    Modified: TP={tp_mod}, FP={fp_mod}, FN={fn_mod} | Prec={prec_mod:.4f}, Rec={rec_mod:.4f}")
    print(f"    Removed:  TP={tp_rem}, FP={fp_rem}, FN={fn_rem} | Prec={prec_rem:.4f}, Rec={rec_rem:.4f}")
    print(f"    Added:    Detected={num_added} (expected 150)")

    assert tp_static == 4851, f"Expected 4851 static matches, got {tp_static}"
    assert tp_mod >= 45, f"Expected >= 45 modified matches, got {tp_mod}"
    assert tp_rem >= 95, f"Expected >= 95 removed matches, got {tp_rem}"
    assert num_added >= 145, f"Expected ~150 added points, got {num_added}"
    assert rec_static == 1.0 and prec_static == 1.0
    assert rec_mod >= 0.90 and prec_mod >= 0.90
    assert rec_rem >= 0.95 and prec_rem >= 0.95

    print(f"  [PASS] test_ground_truth_classification_accuracy (Precision={prec_mod:.2%}, Recall={rec_mod:.2%})")


if __name__ == "__main__":
    print("=" * 70)
    print("Running Mode B: Two-Scene Comparison Test Suite")
    print("=" * 70)
    test_ply_io_and_validation()
    test_identical_scenes_comparison()
    test_different_counts_and_ordering()
    test_coordinate_incompatibility()
    test_ground_truth_classification_accuracy()
    print("=" * 70)
    print("ALL TWO-SCENE COMPARISON TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
