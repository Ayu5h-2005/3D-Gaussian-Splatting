import sys
import time
from pathlib import Path
ROOT = Path(r"D:\3dgs-incremental-research")
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
import gsplat
from src.ply_io import load_3dgs_ply
from src.scene_cache import _fused_tile_invalidation_kernel
from src.scene_comparison import match_gaussians_mutual_knn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ply_a_path = ROOT / "results" / "ply_demo" / "baseline_scene.ply"
ply_b_path = ROOT / "results" / "ply_demo" / "updated_scene.ply"

print("=" * 70)
print("PROFILING TWO-SCENE COMPARISON PIPELINE")
print("=" * 70)

# Warmup GPU
torch.cuda.synchronize()
_ = torch.randn(1000, 1000, device=device) @ torch.randn(1000, 1000, device=device)
torch.cuda.synchronize()

def run_profile_breakdown():
    # 1. PLY Parsing
    t0 = time.perf_counter()
    scene_a = load_3dgs_ply(ply_a_path, device=device)
    t1 = time.perf_counter()
    scene_b = load_3dgs_ply(ply_b_path, device=device)
    t2 = time.perf_counter()
    t_ply_a_ms = (t1 - t0) * 1000.0
    t_ply_b_ms = (t2 - t1) * 1000.0

    # 2. Coordinate compatibility check
    torch.cuda.synchronize()
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    ev_start.record()
    min_a, max_a = scene_a["means"].min(dim=0).values, scene_a["means"].max(dim=0).values
    min_b, max_b = scene_b["means"].min(dim=0).values, scene_b["means"].max(dim=0).values
    center_a = (min_a + max_a) / 2.0
    center_b = (min_b + max_b) / 2.0
    center_dist = torch.norm(center_a - center_b).item()
    diag_a = torch.norm(max_a - min_a).item()
    diag_b = torch.norm(max_b - min_b).item()
    scale_ratio = diag_b / max(diag_a, 1e-6)
    compatible_coords = (center_dist < max(diag_a, diag_b) * 2.0) and (0.1 < scale_ratio < 10.0)
    ev_end.record()
    torch.cuda.synchronize()
    t_compat_ms = ev_start.elapsed_time(ev_end)

    # 3. Nearest-neighbor search & mutual matching breakdown
    pos_a = scene_a["means"]
    pos_b = scene_b["means"]
    na = pos_a.shape[0]
    nb = pos_b.shape[0]
    chunk_size = 4000

    # Measure Forward NN
    ev_start.record()
    nearest_b_for_a = torch.empty(na, dtype=torch.long, device=device)
    dist_b_for_a = torch.empty(na, dtype=torch.float32, device=device)
    for i in range(0, na, chunk_size):
        chunk = pos_a[i:i + chunk_size]
        dists = torch.cdist(chunk, pos_b)
        m_dist, m_idx = torch.min(dists, dim=-1)
        nearest_b_for_a[i:i + chunk_size] = m_idx
        dist_b_for_a[i:i + chunk_size] = m_dist
    ev_end.record()
    torch.cuda.synchronize()
    t_fwd_nn_ms = ev_start.elapsed_time(ev_end)

    # Measure Reverse NN
    ev_start.record()
    nearest_a_for_b = torch.empty(nb, dtype=torch.long, device=device)
    dist_a_for_b = torch.empty(nb, dtype=torch.float32, device=device)
    for j in range(0, nb, chunk_size):
        chunk = pos_b[j:j + chunk_size]
        dists = torch.cdist(chunk, pos_a)
        m_dist, m_idx = torch.min(dists, dim=-1)
        nearest_a_for_b[j:j + chunk_size] = m_idx
        dist_a_for_b[j:j + chunk_size] = m_dist
    ev_end.record()
    torch.cuda.synchronize()
    t_rev_nn_ms = ev_start.elapsed_time(ev_end)

    # Measure Mutual Validation
    ev_start.record()
    is_mutual_a = (nearest_a_for_b[nearest_b_for_a] == torch.arange(na, device=device))
    is_mutual_b = (nearest_b_for_a[nearest_a_for_b] == torch.arange(nb, device=device))
    pos_matched_a = is_mutual_a & (dist_b_for_a <= 0.03)
    pos_matched_b = is_mutual_b & (dist_a_for_b <= 0.03)
    ev_end.record()
    torch.cuda.synchronize()
    t_mutual_ms = ev_start.elapsed_time(ev_end)

    # Measure Attribute distance calculation
    ev_start.record()
    matched_b = nearest_b_for_a
    col_delta_a = torch.norm(scene_a["colors"] - scene_b["colors"][matched_b], dim=-1)
    opac_delta_a = (scene_a["opacities"] - scene_b["opacities"][matched_b]).abs()
    scale_delta_a = torch.norm(scene_a["scales"] - scene_b["scales"][matched_b], dim=-1)
    feat_matched_a = (col_delta_a <= 0.08) & (opac_delta_a <= 0.08) & (scale_delta_a <= 0.08)

    matched_a = nearest_a_for_b
    col_delta_b = torch.norm(scene_b["colors"] - scene_a["colors"][matched_a], dim=-1)
    opac_delta_b = (scene_b["opacities"] - scene_a["opacities"][matched_a]).abs()
    scale_delta_b = torch.norm(scene_b["scales"] - scene_a["scales"][matched_a], dim=-1)
    feat_matched_b = (col_delta_b <= 0.08) & (opac_delta_b <= 0.08) & (scale_delta_b <= 0.08)
    ev_end.record()
    torch.cuda.synchronize()
    t_attr_dist_ms = ev_start.elapsed_time(ev_end)

    # Measure Classification masks
    ev_start.record()
    static_a_mask = pos_matched_a & feat_matched_a
    modified_a_mask = pos_matched_a & (~feat_matched_a)
    removed_a_mask = ~pos_matched_a
    dirty_a_mask = ~static_a_mask

    static_b_mask = pos_matched_b & feat_matched_b
    modified_b_mask = pos_matched_b & (~feat_matched_b)
    added_b_mask = ~pos_matched_b
    dirty_b_mask = ~static_b_mask
    ev_end.record()
    torch.cuda.synchronize()
    t_classify_ms = ev_start.elapsed_time(ev_end)

    # Measure Synchronization / .item()
    t_sync0 = time.perf_counter()
    num_static = int(static_a_mask.sum().item())
    num_modified = int(modified_a_mask.sum().item())
    num_removed = int(removed_a_mask.sum().item())
    num_added = int(added_b_mask.sum().item())
    t_sync_ms = (time.perf_counter() - t_sync0) * 1000.0

    # Camera setup for projection
    viewmats = torch.eye(4, device=device)[None, :, :]
    Ks = torch.tensor([[512.0, 0, 256.0], [0, 512.0, 256.0], [0, 0, 1.0]], device=device)[None, :, :]
    tile_w = (512 + 16 - 1) // 16
    tile_h = (512 + 16 - 1) // 16
    y_grid = torch.arange(tile_h, device=device).view(tile_h, 1)
    x_grid = torch.arange(tile_w, device=device).view(1, tile_w)

    dirty_a_idx = torch.where(dirty_a_mask)[0]
    dirty_b_idx = torch.where(dirty_b_mask)[0]

    # Measure Screen-space projection
    ev_start.record()
    rad_a, m2d_a, dep_a, _, _ = gsplat.fully_fused_projection(
        scene_a["means"][dirty_a_idx], None, scene_a["quats"][dirty_a_idx], scene_a["scales"][dirty_a_idx],
        viewmats, Ks, 512, 512, packed=False
    )
    rad_b, m2d_b, dep_b, _, _ = gsplat.fully_fused_projection(
        scene_b["means"][dirty_b_idx], None, scene_b["quats"][dirty_b_idx], scene_b["scales"][dirty_b_idx],
        viewmats, Ks, 512, 512, packed=False
    )
    m2d_all = torch.cat([m2d_a[0], m2d_b[0]], dim=0)
    rad_all = torch.cat([rad_a[0], rad_b[0]], dim=0)
    dep_all = torch.cat([dep_a[0], dep_b[0]], dim=0)
    ev_end.record()
    torch.cuda.synchronize()
    t_proj_ms = ev_start.elapsed_time(ev_end)

    # Measure Tile invalidation kernel
    ev_start.record()
    tile_mask = _fused_tile_invalidation_kernel(
        m2d_all, rad_all, dep_all, y_grid, x_grid, 512, 512, 16, tile_w, tile_h
    )
    ev_end.record()
    torch.cuda.synchronize()
    t_inval_ms = ev_start.elapsed_time(ev_end)

    return {
        "ply_a_ms": t_ply_a_ms,
        "ply_b_ms": t_ply_b_ms,
        "compat_ms": t_compat_ms,
        "fwd_nn_ms": t_fwd_nn_ms,
        "rev_nn_ms": t_rev_nn_ms,
        "mutual_ms": t_mutual_ms,
        "attr_dist_ms": t_attr_dist_ms,
        "classify_ms": t_classify_ms,
        "sync_item_ms": t_sync_ms,
        "proj_ms": t_proj_ms,
        "inval_ms": t_inval_ms,
    }

# 1. First run (Cold)
cold = run_profile_breakdown()

# 2. Warm runs
warm_runs = [run_profile_breakdown() for _ in range(10)]
warm_avg = {}
for k in cold:
    warm_avg[k] = float(np.mean([r[k] for r in warm_runs]))

print("\nSTAGE BREAKDOWN TIMINGS:")
print(f"{'Stage':<35} | {'Cold (1st run) ms':<18} | {'Warm (Avg 10) ms':<18}")
print("-" * 75)
for k in cold:
    print(f"{k:<35} | {cold[k]:<18.3f} | {warm_avg[k]:<18.3f}")

total_cold = sum(cold.values())
total_warm = sum(warm_avg.values())
print("-" * 75)
print(f"{'Total Pipeline Cost':<35} | {total_cold:<18.3f} | {total_warm:<18.3f}")

print("\nNotice PLY Loading vs In-Memory CUDA Processing:")
ply_cold = cold["ply_a_ms"] + cold["ply_b_ms"]
ply_warm = warm_avg["ply_a_ms"] + warm_avg["ply_b_ms"]
gpu_warm = total_warm - ply_warm
print(f"Total PLY parsing (A + B): Cold = {ply_cold:.3f} ms, Warm = {ply_warm:.3f} ms")
print(f"Total In-Memory GPU Math:  Cold = {total_cold - ply_cold:.3f} ms, Warm = {gpu_warm:.3f} ms")
