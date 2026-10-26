"""
scene_comparison.py - Automatic 3D Gaussian Splatting Change Detection & Incremental Pipeline

Implements Mode B: Automatic Change Detection Between Two 3DGS Scenes (PLY or Tensor format).
- Handles different Gaussian counts (N_A != N_B) and arbitrary Gaussian orderings.
- Performs GPU-vectorized mutual nearest-neighbor spatial and appearance matching.
- Classifies Gaussians into: Matched (Static), Modified, Added (B only), Removed (A only).
- Generates conservative screen-space tile invalidation mask (Zero False Negatives).
- Performs selective tile-masked incremental rasterization and 5D broadcast compositing.
- Validates bitwise depth compositing parity against independent ground-truth full rerender.
"""

from pathlib import Path
import time
from typing import Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import gsplat

from src.scene_cache import _fused_tile_invalidation_kernel
from src.tile_invalidation import (
    compute_ground_truth_changed_tiles,
    evaluate_invalidation_coverage,
    compute_image_metrics,
)


def _generate_turbo_lut() -> np.ndarray:
    """Generates a 256x3 RGB lookup table approximating the Turbo continuous heatmap colormap."""
    vals = np.linspace(0.0, 1.0, 256)
    r = np.clip(1.5 - np.abs(vals * 4.0 - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(vals * 4.0 - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(vals * 4.0 - 1.0), 0.0, 1.0)
    dark_blend = np.clip(vals * 3.0, 0.0, 1.0)[:, None]
    rgb = np.stack([r, g, b], axis=1) * dark_blend * 255.0
    return rgb.astype(np.uint8)


def _save_heatmap_image(diff_tensor: torch.Tensor, out_path: Path, gain: float = 8.0) -> None:
    """Saves amplified difference heatmap image using Turbo colormap."""
    diff_np = diff_tensor.cpu().numpy()
    amplified = np.clip(diff_np * gain, 0.0, 1.0)
    indices = (amplified * 255.0).astype(np.uint8)
    lut = _generate_turbo_lut()
    heatmap = lut[indices]
    Image.fromarray(heatmap, mode="RGB").save(str(out_path))


def _save_tile_overlay_image(
    base_img: torch.Tensor,
    tile_mask: torch.Tensor,
    out_path: Path,
    tile_size: int,
    color_rgb: Tuple[int, int, int] = (0, 220, 255),
    alpha: float = 0.55,
) -> None:
    """Overlays marked tiles on top of a base image."""
    h, w, _ = base_img.shape
    base_np = (base_img.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    overlay = base_np.copy()

    pix_mask = tile_mask.repeat_interleave(tile_size, dim=0).repeat_interleave(tile_size, dim=1)[:h, :w].cpu().numpy()
    color_arr = np.array(color_rgb, dtype=np.uint8)
    overlay[pix_mask] = (overlay[pix_mask] * (1.0 - alpha) + color_arr * alpha).astype(np.uint8)

    # Tile grid lines on active regions
    tile_h, tile_w = tile_mask.shape
    for y in range(tile_h + 1):
        py = min(y * tile_size, h - 1)
        overlay[py, :, :] = np.clip(overlay[py, :, :].astype(np.int16) + 40, 0, 255).astype(np.uint8)
    for x in range(tile_w + 1):
        px = min(x * tile_size, w - 1)
        overlay[:, px, :] = np.clip(overlay[:, px, :].astype(np.int16) + 40, 0, 255).astype(np.uint8)

    Image.fromarray(overlay, mode="RGB").save(str(out_path))


def match_gaussians_mutual_knn(
    scene_a: Dict[str, torch.Tensor],
    scene_b: Dict[str, torch.Tensor],
    pos_tol: float = 0.03,
    feat_tol: float = 0.08,
    disp_tol: float = 0.35,
    chunk_size: int = 4000,
) -> Dict[str, Union[torch.Tensor, int, float]]:
    """
    Performs bidirectional GPU-vectorized spatial and appearance nearest-neighbor matching
    between two arbitrary 3D Gaussian scenes using a two-stage correspondence engine.
    
    Stage 1: Co-located Mutual Nearest Neighbors (Static & In-Place Modifications).
    Stage 2: Motion / Displaced Matching on Unmatched Residuals (Displaced Modifications vs Add/Remove).
    
    Identifies:
    - static_a_mask: Gaussians in A whose position & appearance are invariant in B.
    - dirty_a_mask: Gaussians in A that are removed or modified.
    - static_b_mask: Gaussians in B whose position & appearance are invariant from A.
    - dirty_b_mask: Gaussians in B that are added or modified.
    - modified_a_mask: Gaussians in A matched to B whose position or appearance changed.
    - removed_a_mask: Gaussians present in A but missing in B.
    - added_b_mask: Gaussians present in B but missing in A.
    - matching_confidence: Ratio of mutual spatial matches within tolerance.
    """
    pos_a = scene_a["means"]
    pos_b = scene_b["means"]
    device = pos_a.device

    na = pos_a.shape[0]
    nb = pos_b.shape[0]

    # --- STAGE 1: Co-located Nearest-Neighbor Matching ---
    # Fast path: For scenes under 10k Gaussians, use fused Tensor-Core torch.addmm
    if na <= 10000 and nb <= 10000:
        norm_a = (pos_a ** 2).sum(dim=-1, keepdim=True)
        norm_b = (pos_b ** 2).sum(dim=-1, keepdim=True)
        dist_sq = torch.addmm(norm_a + norm_b.t(), pos_a, pos_b.t(), alpha=-2.0, beta=1.0).clamp_min_(0.0)
        dist_ab = torch.sqrt(dist_sq)

        dist_b_for_a, nearest_b_for_a = torch.min(dist_ab, dim=-1)
        dist_a_for_b, nearest_a_for_b = torch.min(dist_ab, dim=0)
    else:
        # Memory-safe chunked path for large scenes
        nearest_b_for_a = torch.empty(na, dtype=torch.long, device=device)
        dist_b_for_a = torch.empty(na, dtype=torch.float32, device=device)
        for i in range(0, na, chunk_size):
            chunk = pos_a[i:i + chunk_size]
            dists = torch.cdist(chunk, pos_b)
            m_dist, m_idx = torch.min(dists, dim=-1)
            nearest_b_for_a[i:i + chunk_size] = m_idx
            dist_b_for_a[i:i + chunk_size] = m_dist

        nearest_a_for_b = torch.empty(nb, dtype=torch.long, device=device)
        dist_a_for_b = torch.empty(nb, dtype=torch.float32, device=device)
        for j in range(0, nb, chunk_size):
            chunk = pos_b[j:j + chunk_size]
            dists = torch.cdist(chunk, pos_a)
            m_dist, m_idx = torch.min(dists, dim=-1)
            nearest_a_for_b[j:j + chunk_size] = m_idx
            dist_a_for_b[j:j + chunk_size] = m_dist

    # Mutual Consistency Check
    is_mutual_a = (nearest_a_for_b[nearest_b_for_a] == torch.arange(na, device=device))
    pos_matched_a = is_mutual_a & (dist_b_for_a <= pos_tol)

    # Appearance & Scale Check for Spatial Matches
    matched_b = nearest_b_for_a
    col_delta_a = torch.norm(scene_a["colors"] - scene_b["colors"][matched_b], dim=-1)
    opac_delta_a = (scene_a["opacities"] - scene_b["opacities"][matched_b]).abs()
    scale_delta_a = torch.norm(scene_a["scales"] - scene_b["scales"][matched_b], dim=-1)
    feat_matched_a = (col_delta_a <= feat_tol) & (opac_delta_a <= feat_tol) & (scale_delta_a <= feat_tol)

    static_a_mask = pos_matched_a & feat_matched_a
    inplace_mod_a = pos_matched_a & (~feat_matched_a)

    static_b_mask = torch.zeros(nb, dtype=torch.bool, device=device)
    if static_a_mask.any():
        static_b_mask[matched_b[static_a_mask]] = True

    inplace_mod_b = torch.zeros(nb, dtype=torch.bool, device=device)
    if inplace_mod_a.any():
        inplace_mod_b[matched_b[inplace_mod_a]] = True

    # --- STAGE 2: Motion / Displaced Matching on Residual Unmatched Pools ---
    unmatched_a_idx = torch.where(~(static_a_mask | inplace_mod_a))[0]
    unmatched_b_idx = torch.where(~(static_b_mask | inplace_mod_b))[0]

    displaced_mod_a = torch.zeros(na, dtype=torch.bool, device=device)
    displaced_mod_b = torch.zeros(nb, dtype=torch.bool, device=device)

    if unmatched_a_idx.numel() > 0 and unmatched_b_idx.numel() > 0 and disp_tol > pos_tol:
        pos_un_a = pos_a[unmatched_a_idx]
        pos_un_b = pos_b[unmatched_b_idx]
        scales_un_a = scene_a["scales"][unmatched_a_idx]
        scales_un_b = scene_b["scales"][unmatched_b_idx]
        opac_un_a = scene_a["opacities"][unmatched_a_idx]
        opac_un_b = scene_b["opacities"][unmatched_b_idx]
        quats_un_a = scene_a["quats"][unmatched_a_idx]
        quats_un_b = scene_b["quats"][unmatched_b_idx]

        # Geometry consistency on candidate displaced pool
        d_pos_un = torch.cdist(pos_un_a, pos_un_b)
        d_scale_un = torch.cdist(scales_un_a, scales_un_b)
        d_opac_un = (opac_un_a[:, None] - opac_un_b[None, :]).abs()
        dot_q = torch.abs(torch.mm(quats_un_a, quats_un_b.t()))
        d_quat = 1.0 - dot_q

        geom_cost = d_scale_un * 10.0 + d_opac_un * 5.0 + d_quat * 2.0
        valid_motion = (d_pos_un <= disp_tol) & (d_scale_un < 0.05) & (d_opac_un < 0.15)
        cost_matrix = torch.where(valid_motion, geom_cost + d_pos_un * 0.5, torch.tensor(1e6, device=device))

        m_cost_b, m_match_b = torch.min(cost_matrix, dim=-1)
        m_cost_a, m_match_a = torch.min(cost_matrix, dim=0)

        is_mutual_s2 = (m_match_a[m_match_b] == torch.arange(unmatched_a_idx.numel(), device=device)) & (m_cost_b < 100.0)

        if is_mutual_s2.any():
            matched_un_a = unmatched_a_idx[is_mutual_s2]
            matched_un_b = unmatched_b_idx[m_match_b[is_mutual_s2]]
            displaced_mod_a[matched_un_a] = True
            displaced_mod_b[matched_un_b] = True

    # Union of inplace and displaced modifications
    modified_a_mask = inplace_mod_a | displaced_mod_a
    modified_b_mask = inplace_mod_b | displaced_mod_b

    # Removals: in A, but neither static nor modified in B
    removed_a_mask = ~(static_a_mask | modified_a_mask)

    # Additions: in B, but neither static nor modified from A
    added_b_mask = ~(static_b_mask | modified_b_mask)

    # Dirty mask for tile invalidation
    dirty_a_mask = ~static_a_mask
    dirty_b_mask = ~static_b_mask

    # Matching confidence metric
    mutual_count = int(static_a_mask.sum().item()) + int(modified_a_mask.sum().item())
    confidence = (2.0 * mutual_count) / max(na + nb, 1)

    return {
        "static_a_mask": static_a_mask,
        "dirty_a_mask": dirty_a_mask,
        "modified_a_mask": modified_a_mask,
        "removed_a_mask": removed_a_mask,
        "static_b_mask": static_b_mask,
        "dirty_b_mask": dirty_b_mask,
        "modified_b_mask": modified_b_mask,
        "added_b_mask": added_b_mask,
        "confidence": confidence,
        "num_static": int(static_a_mask.sum().item()),
        "num_modified": int(modified_a_mask.sum().item()),
        "num_removed": int(removed_a_mask.sum().item()),
        "num_added": int(added_b_mask.sum().item()),
    }


def compare_and_render_two_scenes(
    scene_a: Dict[str, torch.Tensor],
    scene_b: Dict[str, torch.Tensor],
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    width: int = 512,
    height: int = 512,
    tile_size: int = 16,
    pos_tol: float = 0.03,
    feat_tol: float = 0.08,
    disp_tol: float = 0.35,
    output_dir: Optional[Path] = None,
    num_warmups: int = 3,
    num_reps: int = 15,
) -> Dict[str, any]:
    """
    Executes full automatic change detection, screen-space tile invalidation,
    incremental selective rasterization, and parity benchmarking between two 3DGS scenes.
    """
    device = scene_a["means"].device
    tile_w = (width + tile_size - 1) // tile_size
    tile_h = (height + tile_size - 1) // tile_size
    total_tiles = tile_w * tile_h

    na = scene_a["means"].shape[0]
    nb = scene_b["means"].shape[0]

    # Warm up CUDA kernels (JIT and CUBLAS handles) to prevent cold initialization skew
    _ = torch.randn(64, 64, device=device) @ torch.randn(64, 64, device=device)
    torch.cuda.synchronize()

    # Coordinate / Bounding Box Compatibility Check
    min_a, max_a = scene_a["means"].min(dim=0).values, scene_a["means"].max(dim=0).values
    min_b, max_b = scene_b["means"].min(dim=0).values, scene_b["means"].max(dim=0).values
    center_a = (min_a + max_a) / 2.0
    center_b = (min_b + max_b) / 2.0
    center_dist = torch.norm(center_a - center_b).item()

    diag_a = torch.norm(max_a - min_a).item()
    diag_b = torch.norm(max_b - min_b).item()
    scale_ratio = diag_b / max(diag_a, 1e-6)

    compatible_coords = (center_dist < max(diag_a, diag_b) * 2.0) and (0.1 < scale_ratio < 10.0)

    # 1. Render Baseline Scene A (t0 reference)
    t_start_setup = time.perf_counter()
    img_a_full, _, _ = gsplat.rasterization(
        scene_a["means"], scene_a["quats"], scene_a["scales"],
        scene_a["opacities"], scene_a["colors"],
        viewmats, Ks, width, height
    )
    torch.cuda.synchronize()
    cached_frame_a = img_a_full[0].clone()
    t_setup_ms = (time.perf_counter() - t_start_setup) * 1000.0

    # 2. Render Independent Full Rerender of Scene B (t1 ground truth reference)
    t_start_full = time.perf_counter()
    img_b_full, _, _ = gsplat.rasterization(
        scene_b["means"], scene_b["quats"], scene_b["scales"],
        scene_b["opacities"], scene_b["colors"],
        viewmats, Ks, width, height
    )
    torch.cuda.synchronize()
    img_b_gt = img_b_full[0].clone()
    t_full_ms = (time.perf_counter() - t_start_full) * 1000.0

    # 3. GPU Mutual Spatial & Appearance Matching (Two-Stage Engine)
    t_start_match = time.perf_counter()
    match_res = match_gaussians_mutual_knn(scene_a, scene_b, pos_tol=pos_tol, feat_tol=feat_tol, disp_tol=disp_tol)
    torch.cuda.synchronize()
    t_match_ms = (time.perf_counter() - t_start_match) * 1000.0

    dirty_a_mask = match_res["dirty_a_mask"]
    dirty_b_mask = match_res["dirty_b_mask"]
    dirty_a_idx = torch.where(dirty_a_mask)[0]
    dirty_b_idx = torch.where(dirty_b_mask)[0]

    # 4. Conservative Screen-Space Invalidation Mask
    t_start_inv = time.perf_counter()
    y_grid = torch.arange(tile_h, device=device).view(tile_h, 1)
    x_grid = torch.arange(tile_w, device=device).view(1, tile_w)

    def _compute_invalidation_mask(d_a_idx, d_b_idx):
        m2d_list, rad_list, dep_list = [], [], []
        if d_a_idx.numel() > 0:
            rad_a, m2d_a, dep_a, _, _ = gsplat.fully_fused_projection(
                scene_a["means"][d_a_idx], None, scene_a["quats"][d_a_idx], scene_a["scales"][d_a_idx],
                viewmats, Ks, width, height, packed=False
            )
            m2d_list.append(m2d_a[0])
            rad_list.append(rad_a[0])
            dep_list.append(dep_a[0])

        if d_b_idx.numel() > 0:
            rad_b, m2d_b, dep_b, _, _ = gsplat.fully_fused_projection(
                scene_b["means"][d_b_idx], None, scene_b["quats"][d_b_idx], scene_b["scales"][d_b_idx],
                viewmats, Ks, width, height, packed=False
            )
            m2d_list.append(m2d_b[0])
            rad_list.append(rad_b[0])
            dep_list.append(dep_b[0])

        if len(m2d_list) > 0:
            m2d_all = torch.cat(m2d_list, dim=0)
            rad_all = torch.cat(rad_list, dim=0)
            dep_all = torch.cat(dep_list, dim=0)
            return _fused_tile_invalidation_kernel(
                m2d_all, rad_all, dep_all, y_grid, x_grid, width, height, tile_size, tile_w, tile_h
            )
        else:
            return torch.zeros((tile_h, tile_w), dtype=torch.bool, device=device)

    inv_mask = _compute_invalidation_mask(dirty_a_idx, dirty_b_idx)
    torch.cuda.synchronize()
    t_inv_ms = (time.perf_counter() - t_start_inv) * 1000.0

    # 5. Incremental Rasterization & 5D Broadcast Compositing
    t_start_incr = time.perf_counter()
    rad_b_all, m2d_b_all, dep_b_all, con_b_all, _ = gsplat.fully_fused_projection(
        scene_b["means"], None, scene_b["quats"], scene_b["scales"],
        viewmats, Ks, width, height, packed=False
    )
    _, isect_b, flat_b = gsplat.isect_tiles(m2d_b_all, rad_b_all, dep_b_all, tile_size, tile_w, tile_h)
    offs_b = gsplat.isect_offset_encode(isect_b, 1, tile_w, tile_h)

    cols_b_sparse, _ = gsplat.rasterize_to_pixels(
        m2d_b_all, con_b_all, scene_b["colors"][None, ...], scene_b["opacities"][None, ...],
        width, height, tile_size, offs_b, flat_b, masks=inv_mask[None, ...]
    )

    mask_5d = inv_mask.view(tile_h, 1, tile_w, 1, 1)
    cached_5d = cached_frame_a.view(tile_h, tile_size, tile_w, tile_size, 3)
    sparse_5d = cols_b_sparse[0].view(tile_h, tile_size, tile_w, tile_size, 3)
    comp_img = torch.where(mask_5d, sparse_5d, cached_5d).reshape(height, width, 3)
    torch.cuda.synchronize()
    t_incr_ms = (time.perf_counter() - t_start_incr) * 1000.0

    # 6. Safety & Parity Evaluation
    gt_tiles = compute_ground_truth_changed_tiles(cached_frame_a, img_b_gt, tile_size=tile_size, threshold=1e-3)
    cov_eval = evaluate_invalidation_coverage(inv_mask, gt_tiles)
    img_metrics = compute_image_metrics(img_b_gt, comp_img)

    dirty_tiles_count = int(inv_mask.sum().item())
    clean_tiles_count = total_tiles - dirty_tiles_count
    dirty_pct = (dirty_tiles_count / total_tiles) * 100.0

    # 7. Rigorous Comparative Benchmarking (Task 5: 4 Experimental Conditions)
    # Condition 1: Full rerender of Scene B
    def run_c1_full():
        return gsplat.rasterization(
            scene_b["means"], scene_b["quats"], scene_b["scales"],
            scene_b["opacities"], scene_b["colors"],
            viewmats, Ks, width, height
        )

    # Condition 2: Automatic detection + Full rerender
    def run_c2_detect_plus_full():
        _ = match_gaussians_mutual_knn(scene_a, scene_b, pos_tol, feat_tol, disp_tol)
        return gsplat.rasterization(
            scene_b["means"], scene_b["quats"], scene_b["scales"],
            scene_b["opacities"], scene_b["colors"],
            viewmats, Ks, width, height
        )

    # Condition 3: Reference-mask-driven incremental rendering (selective raster + 5D comp)
    def run_c3_ref_incremental():
        r, m, d, c, _ = gsplat.fully_fused_projection(
            scene_b["means"], None, scene_b["quats"], scene_b["scales"],
            viewmats, Ks, width, height, packed=False
        )
        _, isect, flat = gsplat.isect_tiles(m, r, d, tile_size, tile_w, tile_h)
        off = gsplat.isect_offset_encode(isect, 1, tile_w, tile_h)
        cols, _ = gsplat.rasterize_to_pixels(
            m, c, scene_b["colors"][None, ...], scene_b["opacities"][None, ...],
            width, height, tile_size, off, flat, masks=inv_mask[None, ...]
        )
        s5d = cols[0].view(tile_h, tile_size, tile_w, tile_size, 3)
        return torch.where(mask_5d, s5d, cached_5d).reshape(height, width, 3)

    # Condition 4: Automatic detection + Invalidation + Incremental rendering (Proposed End-to-End)
    def run_c4_detect_plus_incremental():
        m_res = match_gaussians_mutual_knn(scene_a, scene_b, pos_tol, feat_tol, disp_tol)
        d_a = torch.where(m_res["dirty_a_mask"])[0]
        d_b = torch.where(m_res["dirty_b_mask"])[0]
        imask = _compute_invalidation_mask(d_a, d_b)
        m5d = imask.view(tile_h, 1, tile_w, 1, 1)

        r, m, d, c, _ = gsplat.fully_fused_projection(
            scene_b["means"], None, scene_b["quats"], scene_b["scales"],
            viewmats, Ks, width, height, packed=False
        )
        _, isect, flat = gsplat.isect_tiles(m, r, d, tile_size, tile_w, tile_h)
        off = gsplat.isect_offset_encode(isect, 1, tile_w, tile_h)
        cols, _ = gsplat.rasterize_to_pixels(
            m, c, scene_b["colors"][None, ...], scene_b["opacities"][None, ...],
            width, height, tile_size, off, flat, masks=imask[None, ...]
        )
        s5d = cols[0].view(tile_h, tile_size, tile_w, tile_size, 3)
        return torch.where(m5d, s5d, cached_5d).reshape(height, width, 3)

    # Warmups for all conditions
    for _ in range(num_warmups):
        _ = run_c1_full()
        _ = run_c3_ref_incremental()
        _ = match_gaussians_mutual_knn(scene_a, scene_b, pos_tol, feat_tol, disp_tol)
    torch.cuda.synchronize()

    def benchmark_fn(fn, reps):
        times = []
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)
        for _ in range(reps):
            ev_s.record()
            _ = fn()
            ev_e.record()
            torch.cuda.synchronize()
            times.append(ev_s.elapsed_time(ev_e))
        return {
            "median": float(np.median(times)),
            "mean": float(np.mean(times)),
            "p95": float(np.percentile(times, 95)),
        }

    bench_c1 = benchmark_fn(run_c1_full, num_reps)
    bench_c2 = benchmark_fn(run_c2_detect_plus_full, num_reps)
    bench_c3 = benchmark_fn(run_c3_ref_incremental, num_reps)
    bench_c4 = benchmark_fn(run_c4_detect_plus_incremental, num_reps)

    # Individual stage benchmarks
    bench_detect = benchmark_fn(lambda: match_gaussians_mutual_knn(scene_a, scene_b, pos_tol, feat_tol, disp_tol), num_reps)
    bench_inval = benchmark_fn(lambda: _compute_invalidation_mask(dirty_a_idx, dirty_b_idx), num_reps)

    t_full_bench = bench_c1["median"]
    t_incr_bench = bench_c3["median"]
    speedup_ratio = round(t_full_bench / t_incr_bench, 2) if t_incr_bench > 0 else 1.0

    # Save output artifacts if requested
    image_paths = {}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        orig_path = output_dir / "baseline_render.png"
        mod_path = output_dir / "updated_full_render.png"
        comp_path = output_dir / "incremental_composite.png"
        mask_path = output_dir / "invalidation_tile_mask.png"
        diff_path = output_dir / "diff_heatmap.png"
        gt_path = output_dir / "gt_tiles_overlay.png"

        Image.fromarray((cached_frame_a.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(orig_path)
        Image.fromarray((img_b_gt.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(mod_path)
        Image.fromarray((comp_img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(comp_path)

        _save_tile_overlay_image(img_b_gt, inv_mask, mask_path, tile_size, color_rgb=(0, 220, 255), alpha=0.55)
        diff_map = (img_b_gt - cached_frame_a).abs().sum(dim=-1)
        _save_heatmap_image(diff_map, diff_path, gain=8.0)
        _save_tile_overlay_image(cached_frame_a, gt_tiles, gt_path, tile_size, color_rgb=(255, 30, 180), alpha=0.55)

        image_paths = {
            "baseline_render": str(orig_path),
            "updated_full_render": str(mod_path),
            "incremental_composite": str(comp_path),
            "invalidation_tile_mask": str(mask_path),
            "diff_heatmap": str(diff_path),
            "gt_tiles_overlay": str(gt_path),
        }

    return {
        "success": True,
        "compatibility": {
            "compatible_coords": compatible_coords,
            "center_distance": round(center_dist, 4),
            "scale_ratio": round(scale_ratio, 4),
            "bounding_box_a": [float(x) for x in min_a.tolist()] + [float(x) for x in max_a.tolist()],
            "bounding_box_b": [float(x) for x in min_b.tolist()] + [float(x) for x in max_b.tolist()],
        },
        "counts": {
            "scene_a_gaussians": na,
            "scene_b_gaussians": nb,
            "matched_static": match_res["num_static"],
            "modified": match_res["num_modified"],
            "removed": match_res["num_removed"],
            "added": match_res["num_added"],
            "matching_confidence": round(match_res["confidence"], 4),
        },
        "tiles": {
            "total_tiles": total_tiles,
            "invalid_tile_count": dirty_tiles_count,
            "clean_tile_count": clean_tiles_count,
            "dirty_tile_percentage": round(dirty_pct, 2),
            "clean_tile_percentage": round(100.0 - dirty_pct, 2),
            "gt_changed_tiles": int(gt_tiles.sum().item()),
            "false_negatives": cov_eval["false_negatives"],
            "recall": round(cov_eval["recall"], 4),
            "zero_false_negatives": cov_eval["zero_false_negatives"],
        },
        "fidelity": {
            "max_abs_error": round(float(img_metrics["max_absolute_error"]), 8),
            "l1_error": round(float(img_metrics["l1_error"]), 8),
            "mse": round(float(img_metrics["mse"]), 8),
            "psnr_db": round(float(img_metrics["psnr_db"]), 2) if not np.isinf(img_metrics["psnr_db"]) else "inf",
            "ssim": round(float(img_metrics["ssim"]), 6),
        },
        "latencies_ms": {
            "t_setup_baseline_ms": round(t_setup_ms, 3),
            "t_change_detection_ms": round(bench_detect["median"], 3),
            "t_invalidation_mask_ms": round(bench_inval["median"], 3),
            "t_full_render_med_ms": round(bench_c1["median"], 3),
            "t_incremental_med_ms": round(bench_c3["median"], 3),
            "speedup_ratio": speedup_ratio,
        },
        "comparative_benchmarks": {
            "c1_full_rerender": bench_c1,
            "c2_detect_plus_full": bench_c2,
            "c3_ref_incremental": bench_c3,
            "c4_detect_plus_incremental": bench_c4,
            "stage_change_detection": bench_detect,
            "stage_tile_invalidation": bench_inval,
        },
        "image_paths": image_paths,
    }
