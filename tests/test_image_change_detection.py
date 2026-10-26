"""
test_image_change_detection.py - Unit & Correctness Tests for 2D Image Change Detection

Validates:
1. Identical images: zero changed pixels (0.0%), empty mask, fast execution.
2. Synthetic localized change: accurate detection of known geometric modification.
3. Image size mismatch: safe dimension alignment/rescaling without crashing.
4. Alignment handling & fallback: ORB/ECC alignment with graceful fallback when unaligned.
5. High threshold / no-change result: clean handling of zero changes.
6. Corrupted / invalid input handling: informative ImageChangeDetectionError.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from src.image_change_detection import (
    load_image_safe,
    align_images_opencv,
    detect_image_changes,
    ImageChangeDetectionError,
    HAS_OPENCV,
)


def test_identical_images():
    """Identical images must yield 0 changed pixels, 0.0% change, and no error."""
    print("Testing identical image comparison...")
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[50:150, 50:150] = [100, 150, 200]
    img[120:200, 120:200] = [200, 100, 50]

    res = detect_image_changes(
        image_original=img,
        image_updated=img.copy(),
        enable_alignment=False,
        difference_threshold=15,
        min_region_size=10,
    )

    metrics = res["metrics"]
    assert metrics["changed_pixels"] == 0, f"Expected 0 changed pixels, got {metrics['changed_pixels']}"
    assert metrics["changed_percentage"] == 0.0
    assert metrics["has_changes"] is False
    assert metrics["num_changed_regions"] == 0
    assert res["images"]["binary_mask"].max() == 0
    print(f"  [PASS] test_identical_images: {metrics['changed_pixels']} changed pixels ({metrics['latency_ms']:.2f} ms)")


def test_synthetic_localized_change():
    """Detects a known synthetic rectangle modification with high recall and precision."""
    print("Testing synthetic localized change detection...")
    base = np.zeros((300, 300, 3), dtype=np.uint8)
    base[:] = (40, 40, 40)
    # Add texture/shapes so it's not a flat image
    base[40:120, 40:120] = (120, 140, 160)

    # Modified version: draw bright red rectangle at (150, 150) -> (210, 210) (area: 60x60 = 3600 px)
    upd = base.copy()
    upd[150:210, 150:210] = (240, 30, 30)

    res = detect_image_changes(
        image_original=base,
        image_updated=upd,
        enable_alignment=False,
        blur_kernel_size=0,
        difference_threshold=20,
        min_region_size=50,
        draw_bounding_boxes=True,
    )

    metrics = res["metrics"]
    expected_area = 60 * 60  # 3600 pixels

    detected_area = metrics["changed_pixels"]
    assert abs(detected_area - expected_area) <= 50, f"Expected ~{expected_area} changed pixels, got {detected_area}"
    assert metrics["has_changes"] is True
    assert metrics["num_changed_regions"] >= 1

    # Verify bounding box encloses the changed region
    if HAS_OPENCV:
        bx, by, bw, bh = res["bounding_boxes"][0]
        assert abs(bx - 150) <= 2
        assert abs(by - 150) <= 2
        assert abs(bw - 60) <= 2
        assert abs(bh - 60) <= 2

    print(f"  [PASS] test_synthetic_localized_change: detected {detected_area}/{expected_area} pixels ({metrics['changed_percentage']:.2f}%)")


def test_image_size_mismatch():
    """Different image dimensions must be handled safely without runtime crashes."""
    print("Testing image size mismatch handling...")
    # Reference is 400x300, Target is 200x150
    orig = np.zeros((300, 400, 3), dtype=np.uint8)
    orig[50:150, 50:150] = (200, 100, 80)

    upd_small = np.zeros((150, 200, 3), dtype=np.uint8)
    upd_small[25:75, 25:75] = (200, 100, 80)
    # Add a change on small image
    upd_small[80:120, 80:120] = (0, 255, 100)

    res = detect_image_changes(
        image_original=orig,
        image_updated=upd_small,
        enable_alignment=False,
        difference_threshold=25,
        min_region_size=20,
    )

    assert res["success"] is True
    assert res["dimension_warning"] is not None
    assert "Dimension mismatch" in res["dimension_warning"]
    # Output images must match reference dimensions
    assert res["images"]["highlighted"].shape == (300, 400, 3)
    assert res["images"]["binary_mask"].shape == (300, 400)
    assert res["metrics"]["has_changes"] is True
    print(f"  [PASS] test_image_size_mismatch: {res['dimension_warning']}")


def test_alignment_fallback():
    """Validates alignment execution and graceful fallback when features cannot be matched."""
    print("Testing alignment and fallback...")
    # Flat image with minimal features where feature alignment should gracefully fall back
    img1 = np.ones((200, 200, 3), dtype=np.uint8) * 128
    img2 = np.ones((200, 200, 3), dtype=np.uint8) * 128
    img2[50:80, 50:80] = 200

    aligned, info = align_images_opencv(img1, img2, method="orb")
    # Even if feature detection finds few keypoints, it must return an array without raising
    assert aligned.shape == img1.shape
    assert isinstance(info["success"], bool)
    if not info["success"]:
        assert info["warning"] is not None
        print(f"  [PASS] Alignment fallback triggered gracefully: {info['warning']}")
    else:
        print("  [PASS] Alignment succeeded")

    # Complete detection with alignment enabled must not crash
    res = detect_image_changes(
        image_original=img1,
        image_updated=img2,
        enable_alignment=True,
        alignment_method="orb",
    )
    assert res["success"] is True
    assert res["metrics"]["has_changes"] is True
    print("  [PASS] test_alignment_fallback: complete detection succeeded")


def test_corrupt_and_edge_cases():
    """Checks invalid inputs, corrupt bytes, and extreme thresholds."""
    print("Testing edge cases and error handling...")
    # Empty byte buffer
    try:
        load_image_safe(b"")
        assert False, "Should have raised ImageChangeDetectionError on empty bytes"
    except ImageChangeDetectionError:
        pass

    # Corrupt garbage bytes
    try:
        load_image_safe(b"NOT_A_VALID_IMAGE_HEADER_12345")
        assert False, "Should have raised ImageChangeDetectionError on corrupt bytes"
    except ImageChangeDetectionError:
        pass

    # Extreme threshold (no changes should be detected)
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    upd = np.clip(img.astype(np.int16) + 10, 0, 255).astype(np.uint8)
    res = detect_image_changes(img, upd, difference_threshold=200, enable_alignment=False)
    assert res["metrics"]["changed_pixels"] == 0
    assert res["metrics"]["has_changes"] is False

    print("  [PASS] test_corrupt_and_edge_cases: all edge cases handled safely")


if __name__ == "__main__":
    print("=" * 70)
    print("Running Image Change Detection Test Suite")
    print("=" * 70)
    test_identical_images()
    test_synthetic_localized_change()
    test_image_size_mismatch()
    test_alignment_fallback()
    test_corrupt_and_edge_cases()
    print("=" * 70)
    print("ALL IMAGE CHANGE DETECTION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
