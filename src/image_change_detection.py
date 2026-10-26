"""
image_change_detection.py - 2D Image-Space Change Detection & Alignment Pipeline

Provides robust 2D pixel-space change detection between two images:
- Handles images of different dimensions safely.
- Optional camera motion alignment via OpenCV (ORB feature matching or ECC) with graceful fallback.
- Color space conversion (Grayscale, RGB Euclidean, CIELAB delta E).
- Sensor noise reduction via Gaussian blur.
- Difference map calculation and configurable thresholding.
- Minimum connected-region area filtering and morphological cleanup.
- Visualizations: aligned overlay, normalized Turbo difference heatmap, and highlighted output.
- Precision metrics: changed pixel counts, percentage, cluster counts, and processing latency.

NOTE: This performs 2D pixel-space image change detection. It does not reconstruct
3D Gaussian Splatting parameters, identify changed Gaussians, or benchmark incremental rasterization.
"""

import io
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageFilter

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    cv2 = None
    HAS_OPENCV = False


class ImageChangeDetectionError(Exception):
    """Raised when image loading or processing fails."""
    pass


def _generate_turbo_lut() -> np.ndarray:
    """Generates a 256x3 RGB lookup table approximating the Turbo colormap."""
    vals = np.linspace(0.0, 1.0, 256)
    r = np.clip(1.5 - np.abs(vals * 4.0 - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(vals * 4.0 - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(vals * 4.0 - 1.0), 0.0, 1.0)
    dark_blend = np.clip(vals * 3.0, 0.0, 1.0)[:, None]
    rgb = np.stack([r, g, b], axis=1) * dark_blend * 255.0
    return rgb.astype(np.uint8)


def load_image_safe(
    source: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray],
    max_dimension: int = 4096,
) -> np.ndarray:
    """
    Loads an image from various sources safely and converts to an RGB uint8 NumPy array.
    
    Args:
        source: File path, byte stream, PIL Image, or NumPy array.
        max_dimension: Maximum allowed width or height to prevent OOM.
        
    Returns:
        np.ndarray: uint8 RGB image array of shape (H, W, 3).
    """
    try:
        if isinstance(source, np.ndarray):
            if source.ndim == 2:
                # Grayscale to RGB
                arr = np.stack([source] * 3, axis=-1)
            elif source.ndim == 3:
                if source.shape[2] == 4:
                    # RGBA to RGB
                    arr = source[:, :, :3]
                elif source.shape[2] == 3:
                    arr = source
                elif source.shape[2] == 1:
                    arr = np.repeat(source, 3, axis=-1)
                else:
                    raise ImageChangeDetectionError(f"Unsupported channel count: {source.shape[2]}")
            else:
                raise ImageChangeDetectionError(f"Invalid array dimensions: {source.ndim}")

            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)
            return arr.copy()

        if isinstance(source, Image.Image):
            pil_img = source.convert("RGB")
        elif isinstance(source, (bytes, bytearray)):
            if len(source) == 0:
                raise ImageChangeDetectionError("Image byte buffer is empty.")
            pil_img = Image.open(io.BytesIO(source)).convert("RGB")
        elif isinstance(source, io.BytesIO):
            source.seek(0)
            pil_img = Image.open(source).convert("RGB")
        elif isinstance(source, (str, Path)):
            p = Path(source)
            if not p.exists():
                raise ImageChangeDetectionError(f"Image file does not exist: {p}")
            pil_img = Image.open(str(p)).convert("RGB")
        else:
            raise ImageChangeDetectionError(f"Unsupported image input type: {type(source)}")

        w, h = pil_img.size
        if w <= 0 or h <= 0:
            raise ImageChangeDetectionError(f"Invalid image dimensions: {w}x{h}")
        if w > max_dimension or h > max_dimension:
            # Rescale preserving aspect ratio
            pil_img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

        return np.array(pil_img, dtype=np.uint8)

    except Exception as e:
        if isinstance(e, ImageChangeDetectionError):
            raise
        raise ImageChangeDetectionError(f"Failed to decode image: {str(e)}") from e


def align_images_opencv(
    img_ref: np.ndarray,
    img_target: np.ndarray,
    method: str = "orb",
    max_features: int = 1500,
) -> Tuple[np.ndarray, Dict[str, Union[bool, str, Optional[float]]]]:
    """
    Aligns img_target to img_ref using OpenCV feature matching or ECC optimization.
    
    Returns:
        aligned_target: np.ndarray of shape (H_ref, W_ref, 3).
        info: Dictionary detailing alignment success, method, and diagnostics.
    """
    h_ref, w_ref = img_ref.shape[:2]
    h_tgt, w_tgt = img_target.shape[:2]

    # Pre-resize target if dimensions differ prior to feature matching
    if (h_tgt, w_tgt) != (h_ref, w_ref):
        if HAS_OPENCV:
            target_resized = cv2.resize(img_target, (w_ref, h_ref), interpolation=cv2.INTER_LINEAR)
        else:
            target_resized = np.array(
                Image.fromarray(img_target).resize((w_ref, h_ref), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
    else:
        target_resized = img_target.copy()

    if not HAS_OPENCV:
        return target_resized, {
            "success": False,
            "method": "none",
            "warning": "OpenCV is not installed; skipped geometric camera alignment.",
            "num_matches": 0,
        }

    try:
        gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY)
        gray_tgt = cv2.cvtColor(target_resized, cv2.COLOR_RGB2GRAY)

        if method.lower() == "ecc":
            # Enhanced Correlation Coefficient maximization (Affine transformation)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
            try:
                _, warp_matrix = cv2.findTransformECC(
                    gray_ref, gray_tgt, warp_matrix, cv2.MOTION_AFFINE, criteria
                )
                aligned = cv2.warpAffine(
                    target_resized, warp_matrix, (w_ref, h_ref),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REPLICATE
                )
                return aligned, {
                    "success": True,
                    "method": "ecc_affine",
                    "warning": None,
                    "num_matches": 0,
                }
            except cv2.error as e:
                # ECC divergence fallback
                return target_resized, {
                    "success": False,
                    "method": "ecc_failed",
                    "warning": f"ECC alignment diverged ({e.msg if hasattr(e, 'msg') else 'gradient stall'}); using unaligned image.",
                    "num_matches": 0,
                }

        else:
            # Default: ORB feature-based homography with RANSAC
            orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=10)
            kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)
            kp_tgt, des_tgt = orb.detectAndCompute(gray_tgt, None)

            if des_ref is None or des_tgt is None or len(kp_ref) < 8 or len(kp_tgt) < 8:
                return target_resized, {
                    "success": False,
                    "method": "orb_insufficient_features",
                    "warning": f"Insufficient keypoints detected (Ref: {len(kp_ref) if des_ref is not None else 0}, Target: {len(kp_tgt) if des_tgt is not None else 0}); continuing without alignment.",
                    "num_matches": 0,
                }

            # Brute-force Hamming distance matcher with k-NN ratio test
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des_ref, des_tgt, k=2)

            # Lowe's ratio test
            good_matches = []
            for m_pair in matches:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < 0.80 * n.distance:
                        good_matches.append(m)

            if len(good_matches) < 8:
                return target_resized, {
                    "success": False,
                    "method": "orb_few_matches",
                    "warning": f"Only {len(good_matches)} good feature matches found (requires ≥ 8 for stable homography); continuing without alignment.",
                    "num_matches": len(good_matches),
                }

            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_tgt[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Compute Homography with RANSAC
            H, inlier_mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 4.0)

            if H is None:
                return target_resized, {
                    "success": False,
                    "method": "orb_ransac_failed",
                    "warning": "RANSAC failed to find a valid planar homography matrix; continuing without alignment.",
                    "num_matches": len(good_matches),
                }

            # Validate homography determinant to catch extreme warp distortions
            det = np.linalg.det(H[:2, :2])
            if det <= 0.2 or det >= 5.0 or np.abs(H[2, 0]) > 0.005 or np.abs(H[2, 1]) > 0.005:
                return target_resized, {
                    "success": False,
                    "method": "orb_unstable_homography",
                    "warning": f"Homography matrix has unstable determinant ({det:.3f}); rejected to prevent severe distortion. Continuing without alignment.",
                    "num_matches": len(good_matches),
                }

            aligned = cv2.warpPerspective(
                target_resized, H, (w_ref, h_ref),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )

            num_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
            return aligned, {
                "success": True,
                "method": "orb_homography",
                "warning": None,
                "num_matches": len(good_matches),
                "num_inliers": num_inliers,
            }

    except Exception as e:
        return target_resized, {
            "success": False,
            "method": "alignment_exception",
            "warning": f"Alignment encountered an error ({str(e)}); continuing without alignment.",
            "num_matches": 0,
        }


def detect_image_changes(
    image_original: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray],
    image_updated: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray],
    enable_alignment: bool = True,
    alignment_method: str = "orb",
    color_space: str = "grayscale",
    blur_kernel_size: int = 5,
    difference_threshold: int = 25,
    min_region_size: int = 50,
    morph_cleanup: bool = True,
    highlight_color: Tuple[int, int, int] = (255, 30, 80),
    highlight_alpha: float = 0.50,
    draw_bounding_boxes: bool = True,
) -> Dict[str, any]:
    """
    Executes complete 2D image change detection between an original and updated image.
    
    Args:
        image_original: Source baseline image.
        image_updated: Source updated image.
        enable_alignment: Whether to run camera movement compensation via OpenCV.
        alignment_method: 'orb' (feature homography) or 'ecc' (affine correlation).
        color_space: 'grayscale', 'rgb', or 'lab'.
        blur_kernel_size: Gaussian blur kernel size (odd integer, 0 to disable).
        difference_threshold: Pixel delta threshold (1 to 255).
        min_region_size: Minimum connected component area in pixels to retain.
        morph_cleanup: Apply morphological open/close to suppress sensor noise.
        highlight_color: RGB tuple for highlighting changed pixels.
        highlight_alpha: Transparency factor for overlay blend.
        draw_bounding_boxes: Whether to draw bounding rectangles around detected change clusters.
        
    Returns:
        Dict containing diagnostic metrics, mask statistics, and visualization images.
    """
    t_start = time.perf_counter()

    # 1. Load images safely
    img_orig = load_image_safe(image_original)
    img_upd = load_image_safe(image_updated)

    h_orig, w_orig, _ = img_orig.shape
    h_upd, w_upd, _ = img_upd.shape

    dim_mismatch = (h_orig, w_orig) != (h_upd, w_upd)
    dimension_warning = None
    if dim_mismatch:
        dimension_warning = f"Dimension mismatch: Original is {w_orig}x{h_orig}, Updated is {w_upd}x{h_upd}. Updated image was scaled to match."

    # 2. Camera Motion Alignment
    alignment_info = {"success": False, "method": "disabled", "warning": None}
    if enable_alignment and HAS_OPENCV:
        img_upd_aligned, alignment_info = align_images_opencv(
            img_ref=img_orig,
            img_target=img_upd,
            method=alignment_method,
        )
    else:
        # Resize updated image if dimensions differ
        if dim_mismatch:
            if HAS_OPENCV:
                img_upd_aligned = cv2.resize(img_upd, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            else:
                img_upd_aligned = np.array(
                    Image.fromarray(img_upd).resize((w_orig, h_orig), Image.Resampling.BILINEAR),
                    dtype=np.uint8,
                )
        else:
            img_upd_aligned = img_upd.copy()

        if enable_alignment and not HAS_OPENCV:
            alignment_info = {
                "success": False,
                "method": "none",
                "warning": "OpenCV is not installed; continuing with unaligned images.",
            }

    # 3. Color Space Conversion & Difference Computation
    color_space_lower = color_space.lower()

    if color_space_lower == "rgb":
        # Multi-channel Euclidean distance
        if blur_kernel_size > 1:
            k = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
            if HAS_OPENCV:
                f_orig = cv2.GaussianBlur(img_orig, (k, k), 0).astype(np.float32)
                f_upd = cv2.GaussianBlur(img_upd_aligned, (k, k), 0).astype(np.float32)
            else:
                p_orig = Image.fromarray(img_orig).filter(ImageFilter.GaussianBlur(k / 2.0))
                p_upd = Image.fromarray(img_upd_aligned).filter(ImageFilter.GaussianBlur(k / 2.0))
                f_orig = np.array(p_orig, dtype=np.float32)
                f_upd = np.array(p_upd, dtype=np.float32)
        else:
            f_orig = img_orig.astype(np.float32)
            f_upd = img_upd_aligned.astype(np.float32)

        # Per-pixel Euclidean distance across R, G, B channels
        diff_map = np.sqrt(np.sum((f_orig - f_upd) ** 2, axis=-1))

    elif color_space_lower == "lab" and HAS_OPENCV:
        # CIELAB color space (perceptually uniform delta E)
        lab_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_upd = cv2.cvtColor(img_upd_aligned, cv2.COLOR_RGB2LAB).astype(np.float32)

        if blur_kernel_size > 1:
            k = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
            lab_orig = cv2.GaussianBlur(lab_orig, (k, k), 0)
            lab_upd = cv2.GaussianBlur(lab_upd, (k, k), 0)

        # Delta E = sqrt(dL^2 + da^2 + db^2)
        diff_map = np.sqrt(np.sum((lab_orig - lab_upd) ** 2, axis=-1))

    else:
        # Default: Grayscale luminance
        if HAS_OPENCV:
            gray_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY).astype(np.float32)
            gray_upd = cv2.cvtColor(img_upd_aligned, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            # Luminance: 0.299 R + 0.587 G + 0.114 B
            gray_orig = (0.299 * img_orig[:, :, 0] + 0.587 * img_orig[:, :, 1] + 0.114 * img_orig[:, :, 2]).astype(np.float32)
            gray_upd = (0.299 * img_upd_aligned[:, :, 0] + 0.587 * img_upd_aligned[:, :, 1] + 0.114 * img_upd_aligned[:, :, 2]).astype(np.float32)

        if blur_kernel_size > 1:
            k = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
            if HAS_OPENCV:
                gray_orig = cv2.GaussianBlur(gray_orig, (k, k), 0)
                gray_upd = cv2.GaussianBlur(gray_upd, (k, k), 0)
            else:
                p_o = Image.fromarray(gray_orig.astype(np.uint8)).filter(ImageFilter.GaussianBlur(k / 2.0))
                p_u = Image.fromarray(gray_upd.astype(np.uint8)).filter(ImageFilter.GaussianBlur(k / 2.0))
                gray_orig = np.array(p_o, dtype=np.float32)
                gray_upd = np.array(p_u, dtype=np.float32)

        diff_map = np.abs(gray_orig - gray_upd)

    # 4. Thresholding
    thresh_val = float(difference_threshold)
    raw_mask = (diff_map >= thresh_val).astype(np.uint8)

    # 5. Morphological Filtering & Connected Component Pruning
    cleaned_mask = raw_mask.copy()
    bounding_boxes: List[Tuple[int, int, int, int]] = []

    if HAS_OPENCV:
        if morph_cleanup:
            # 3x3 open to remove noise speckles, 5x5 close to solidify clusters
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel_open)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_close)

        # Connected component analysis for minimum region size filtering
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)

        filtered_mask = np.zeros_like(cleaned_mask)
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area >= min_region_size:
                filtered_mask[labels == label_id] = 1
                bx = stats[label_id, cv2.CC_STAT_LEFT]
                by = stats[label_id, cv2.CC_STAT_TOP]
                bw = stats[label_id, cv2.CC_STAT_WIDTH]
                bh = stats[label_id, cv2.CC_STAT_HEIGHT]
                bounding_boxes.append((int(bx), int(by), int(bw), int(bh)))

        binary_mask = filtered_mask
    else:
        # Fallback pure NumPy / SciPy if OpenCV is not installed
        try:
            from scipy.ndimage import label, find_objects
            labeled_array, num_features = label(cleaned_mask)
            slices = find_objects(labeled_array)
            filtered_mask = np.zeros_like(cleaned_mask)
            for idx, slc in enumerate(slices):
                if slc is not None:
                    component = (labeled_array[slc] == (idx + 1))
                    area = int(component.sum())
                    if area >= min_region_size:
                        filtered_mask[slc][component] = 1
                        sy, sx = slc
                        bounding_boxes.append((int(sx.start), int(sy.start), int(sx.stop - sx.start), int(sy.stop - sy.start)))
            binary_mask = filtered_mask
        except ImportError:
            # Pure NumPy fallback: compute single global bounding box if above min_region_size
            binary_mask = cleaned_mask
            if int(np.sum(binary_mask)) >= min_region_size:
                rows = np.any(binary_mask, axis=1)
                cols = np.any(binary_mask, axis=0)
                if np.any(rows) and np.any(cols):
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    bounding_boxes.append((int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)))

    # 6. Compute Statistical Metrics
    total_pixels = h_orig * w_orig
    changed_pixels = int(binary_mask.sum())
    unchanged_pixels = total_pixels - changed_pixels
    changed_pct = (changed_pixels / max(total_pixels, 1)) * 100.0
    unchanged_pct = 100.0 - changed_pct

    # 7. Visualization Artifacts
    # A. Difference Heatmap using Turbo Colormap
    diff_norm = np.clip(diff_map / max(thresh_val * 2.5, 1.0), 0.0, 1.0)
    turbo_lut = _generate_turbo_lut()
    heatmap_indices = (diff_norm * 255.0).astype(np.uint8)
    heatmap_rgb = turbo_lut[heatmap_indices]

    # B. Aligned 50/50 Alpha Blend Overlay (Original + Aligned Target)
    aligned_blend = (img_orig.astype(np.float32) * 0.5 + img_upd_aligned.astype(np.float32) * 0.5).astype(np.uint8)

    # C. Highlighted Output: Updated image with colored mask overlay and bounding boxes
    highlighted_img = img_upd_aligned.copy()
    change_pixels_bool = (binary_mask > 0)

    if np.any(change_pixels_bool):
        color_arr = np.array(highlight_color, dtype=np.float32)
        base_region = highlighted_img[change_pixels_bool].astype(np.float32)
        blended_region = (base_region * (1.0 - highlight_alpha) + color_arr * highlight_alpha).clip(0, 255).astype(np.uint8)
        highlighted_img[change_pixels_bool] = blended_region

        # Draw bounding boxes if requested
        if draw_bounding_boxes:
            if HAS_OPENCV:
                for bx, by, bw, bh in bounding_boxes:
                    # Yellow-cyan bounding boxes
                    cv2.rectangle(highlighted_img, (bx, by), (bx + bw, by + bh), (0, 240, 255), 2)
            else:
                try:
                    from PIL import ImageDraw
                    pil_hl = Image.fromarray(highlighted_img)
                    draw = ImageDraw.Draw(pil_hl)
                    for bx, by, bw, bh in bounding_boxes:
                        draw.rectangle([bx, by, bx + bw, by + bh], outline=(0, 240, 255), width=2)
                    highlighted_img = np.array(pil_hl)
                except Exception:
                    pass

    latency_ms = (time.perf_counter() - t_start) * 1000.0

    return {
        "success": True,
        "metrics": {
            "total_pixels": total_pixels,
            "changed_pixels": changed_pixels,
            "unchanged_pixels": unchanged_pixels,
            "changed_percentage": round(changed_pct, 3),
            "unchanged_percentage": round(unchanged_pct, 3),
            "num_changed_regions": len(bounding_boxes),
            "latency_ms": round(latency_ms, 2),
            "has_changes": changed_pixels > 0,
        },
        "settings": {
            "color_space": color_space,
            "blur_kernel_size": blur_kernel_size,
            "difference_threshold": difference_threshold,
            "min_region_size": min_region_size,
            "enable_alignment": enable_alignment,
            "alignment_method": alignment_method,
        },
        "alignment_info": alignment_info,
        "dimension_warning": dimension_warning,
        "bounding_boxes": bounding_boxes,
        "images": {
            "original": img_orig,
            "updated_aligned": img_upd_aligned,
            "aligned_blend": aligned_blend,
            "diff_heatmap": heatmap_rgb,
            "binary_mask": (binary_mask * 255).astype(np.uint8),
            "highlighted": highlighted_img,
        },
    }
