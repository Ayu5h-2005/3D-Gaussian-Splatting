"""
test_experiment_module.py - Unit & Integration Tests for experiment_3dgs.py

Verifies:
1. Execution of run_3dgs_experiment across different perturbation modes.
2. Generation and valid file sizes of all required visualization artifacts.
3. Mathematical correctness: zero false negatives and exact composite fidelity.
4. CLI subprocess execution and JSON stdout serialization.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.experiment_3dgs import run_3dgs_experiment

SCENE_PATH = ROOT / "results" / "procedural_scene_5k.pt"
TEST_OUT_DIR = ROOT / "results" / "test_experiment_out"
GSPLAT_PYTHON = r"D:\miniconda3\envs\gsplat310\python.exe"


def test_experiment_spatial_cluster():
    """Validates spatial cluster perturbation, invalidation, and image generation."""
    print("Testing run_3dgs_experiment with spatial_cluster (1%)...")
    res = run_3dgs_experiment(
        scene_path=SCENE_PATH,
        output_dir=TEST_OUT_DIR,
        mode="spatial_cluster",
        fraction=0.01,
        delta=(0.12, -0.10, 0.04),
        seed=42,
        tile_size=16,
        num_warmups=2,
        num_reps=5,
    )

    assert res["success"] is True
    assert res["perturbation"]["num_changed_gaussians"] == 50
    assert res["tiles"]["zero_false_negatives"] is True
    assert res["tiles"]["recall"] == 1.0
    assert res["fidelity"]["max_abs_error"] == 0.0
    assert res["fidelity"]["ssim"] == 1.0

    # Verify all expected images exist on disk and have non-zero size
    for key, path_str in res["image_paths"].items():
        p = Path(path_str)
        assert p.exists(), f"Image {key} not found at {path_str}"
        assert p.stat().st_size > 1000, f"Image {key} at {path_str} is unusually small ({p.stat().st_size} bytes)"

    print("  [PASS] test_experiment_spatial_cluster")


def test_experiment_single_gaussian():
    """Validates single-gaussian perturbation."""
    print("Testing run_3dgs_experiment with single gaussian...")
    res = run_3dgs_experiment(
        scene_path=SCENE_PATH,
        output_dir=TEST_OUT_DIR,
        mode="single",
        fraction=0.0,
        delta=(0.10, -0.05, 0.02),
        seed=42,
        tile_size=16,
        num_warmups=2,
        num_reps=5,
    )

    assert res["success"] is True
    assert res["perturbation"]["num_changed_gaussians"] == 1
    assert res["tiles"]["zero_false_negatives"] is True
    assert res["tiles"]["invalid_tile_percentage"] < 5.0
    assert res["fidelity"]["max_abs_error"] == 0.0

    print("  [PASS] test_experiment_single_gaussian")


def test_experiment_cli_subprocess():
    """Validates CLI invocation and JSON serialization via subprocess."""
    print("Testing experiment_3dgs.py via CLI subprocess...")
    cmd = [
        GSPLAT_PYTHON,
        str(ROOT / "src" / "experiment_3dgs.py"),
        "--scene-path", str(SCENE_PATH),
        "--output-dir", str(TEST_OUT_DIR),
        "--mode", "random_subset",
        "--fraction", "0.01",
        "--seed", "99",
        "--warmups", "2",
        "--reps", "5",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"CLI exited with error: {proc.stderr}"
    assert "__JSON_START__" in proc.stdout
    assert "__JSON_END__" in proc.stdout

    raw_json = proc.stdout.split("__JSON_START__")[1].split("__JSON_END__")[0].strip()
    payload = json.loads(raw_json)
    assert payload["success"] is True
    assert payload["perturbation"]["mode"] == "random_subset"
    assert payload["tiles"]["zero_false_negatives"] is True

    print("  [PASS] test_experiment_cli_subprocess")


if __name__ == "__main__":
    print("==================================================================")
    print("Running 3DGS Experiment Module Tests")
    print("==================================================================")
    test_experiment_spatial_cluster()
    test_experiment_single_gaussian()
    test_experiment_cli_subprocess()
    print("==================================================================")
    print("ALL EXPERIMENT MODULE TESTS PASSED!")
    print("==================================================================")
