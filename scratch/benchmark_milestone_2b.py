import sys
from pathlib import Path

ROOT = Path(r"D:\3dgs-incremental-research")
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import gsplat
from src.scene_perturbation import perturb_scene
from src.tile_invalidation import compute_tile_invalidation_mask, composite_with_cache, compute_image_metrics

device = torch.device("cuda:0")
scene = torch.load(ROOT / "results" / "procedural_scene_5k.pt", weights_only=False)
for k, v in scene.items():
    if isinstance(v, torch.Tensor):
        scene[k] = v.to(device)

width, height = int(scene["width"]), int(scene["height"])
tile_size = 16
tile_w, tile_h = width // tile_size, height // tile_size

# 1. Warmup and establish baseline render
colors_t0, _, _ = gsplat.rasterization(
    scene["means"], scene["quats"], scene["scales"],
    scene["opacities"], scene["colors"],
    scene["viewmats"], scene["Ks"], width, height
)
cached_frame = colors_t0[0].clone()

# Cache static projections at t0
rad_static, m2d_static, dep_static, con_static, _ = gsplat.fully_fused_projection(
    scene["means"], None, scene["quats"], scene["scales"],
    scene["viewmats"], scene["Ks"], width, height, packed=False
)
rad_static = rad_static.clone()
m2d_static = m2d_static.clone()
dep_static = dep_static.clone()
con_static = con_static.clone()

# Apply perturbation: 1% spatial cluster
perturbed, changed_idx, _ = perturb_scene(
    scene, fraction=0.01, delta=(0.12, -0.10, 0.04), seed=42, mode="spatial_cluster"
)

# -------------------------------------------------------------
# BENCHMARK 1: Full Rerender Baseline (gsplat.rasterization)
# -------------------------------------------------------------
for _ in range(15):
    _ = gsplat.rasterization(
        perturbed["means"], perturbed["quats"], perturbed["scales"],
        perturbed["opacities"], perturbed["colors"],
        perturbed["viewmats"], perturbed["Ks"], width, height
    )
torch.cuda.synchronize()

reps = 50
starts_full = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
ends_full = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
for i in range(reps):
    starts_full[i].record()
    _ = gsplat.rasterization(
        perturbed["means"], perturbed["quats"], perturbed["scales"],
        perturbed["opacities"], perturbed["colors"],
        perturbed["viewmats"], perturbed["Ks"], width, height
    )
    ends_full[i].record()
torch.cuda.synchronize()
times_full = [starts_full[i].elapsed_time(ends_full[i]) for i in range(reps)]
med_full = np.median(times_full)
std_full = np.std(times_full)
print(f"Full Rerender Baseline: {med_full:.4f} ms ± {std_full:.4f} ms")

# -------------------------------------------------------------
# BENCHMARK 2: Milestone 2B Static/Dynamic Partitioned Pipeline
# -------------------------------------------------------------
def run_partitioned_pipeline():
    # Step 1: Project ONLY dynamic Gaussians
    rad_dyn, m2d_dyn, dep_dyn, con_dyn, _ = gsplat.fully_fused_projection(
        perturbed["means"][changed_idx], None,
        perturbed["quats"][changed_idx], perturbed["scales"][changed_idx],
        perturbed["viewmats"], perturbed["Ks"], width, height, packed=False
    )
    
    # Step 2: Compute invalidation mask using cached old projection and new dynamic projection
    # Old footprint: m2d_static[0, changed_idx], rad_static[0, changed_idx]
    # New footprint: m2d_dyn[0], rad_dyn[0]
    m2d_old_changed = m2d_static[0, changed_idx]
    rad_old_changed = rad_static[0, changed_idx]
    dep_old_changed = dep_static[0, changed_idx]
    
    m2d_all_changed = torch.cat([m2d_old_changed, m2d_dyn[0]], dim=0)
    rad_all_changed = torch.cat([rad_old_changed, rad_dyn[0]], dim=0)
    dep_all_changed = torch.cat([dep_old_changed, dep_dyn[0]], dim=0)
    
    valid_c = (dep_all_changed > 0.01) & (rad_all_changed > 0).all(dim=-1)
    xy_min_c = m2d_all_changed - rad_all_changed
    xy_max_c = m2d_all_changed + rad_all_changed
    in_screen_c = (xy_max_c >= 0).all(dim=-1) & (xy_min_c[:, 0] < width) & (xy_min_c[:, 1] < height)
    active_c = valid_c & in_screen_c
    
    uv_min_c = torch.clamp((xy_min_c / tile_size).floor().long(), 0, tile_w - 1)
    uv_max_c = torch.clamp((xy_max_c / tile_size).floor().long(), 0, tile_h - 1)
    
    y_grid = torch.arange(tile_h, device=device).view(tile_h, 1)
    x_grid = torch.arange(tile_w, device=device).view(1, tile_w)
    Y = ((y_grid >= uv_min_c[:, 1].view(1, -1)) & (y_grid <= uv_max_c[:, 1].view(1, -1))).float() * active_c.view(1, -1).float()
    X = ((x_grid >= uv_min_c[:, 0].view(-1, 1)) & (x_grid <= uv_max_c[:, 0].view(-1, 1))).float()
    invalid_mask = (torch.mm(Y, X) > 0)
    
    # Step 3: Update 2D projection buffer (clone static once or slice update)
    m2d_updated = m2d_static.clone()
    rad_updated = rad_static.clone()
    dep_updated = dep_static.clone()
    con_updated = con_static.clone()
    m2d_updated[0, changed_idx] = m2d_dyn[0]
    rad_updated[0, changed_idx] = rad_dyn[0]
    dep_updated[0, changed_idx] = dep_dyn[0]
    con_updated[0, changed_idx] = con_dyn[0]
    
    # Step 4: Candidate filtering via 2D Integral Image
    rad_u = rad_updated[0]
    m2d_u = m2d_updated[0]
    dep_u = dep_updated[0]
    valid_u = (dep_u > 0.01) & (rad_u[:, 0] > 0) & (rad_u[:, 1] > 0)
    u_min_u = torch.clamp(torch.floor((m2d_u[:, 0] - rad_u[:, 0]) / tile_size).long(), 0, tile_w - 1)
    u_max_u = torch.clamp(torch.floor((m2d_u[:, 0] + rad_u[:, 0]) / tile_size).long(), 0, tile_w - 1)
    v_min_u = torch.clamp(torch.floor((m2d_u[:, 1] - rad_u[:, 1]) / tile_size).long(), 0, tile_h - 1)
    v_max_u = torch.clamp(torch.floor((m2d_u[:, 1] + rad_u[:, 1]) / tile_size).long(), 0, tile_h - 1)

    inv_int = invalid_mask.int()
    prefix = torch.zeros((tile_h + 1, tile_w + 1), dtype=torch.int32, device=device)
    prefix[1:, 1:] = torch.cumsum(torch.cumsum(inv_int, dim=0), dim=1)

    box_dirty = (
        prefix[v_max_u + 1, u_max_u + 1]
        - prefix[v_min_u, u_max_u + 1]
        - prefix[v_max_u + 1, u_min_u]
        + prefix[v_min_u, u_min_u]
    )
    touches_dirty = (box_dirty > 0) & valid_u
    cand_idx = torch.nonzero(touches_dirty).squeeze(-1)
    
    # Step 5: Sort only candidates
    m2d_cand = m2d_updated[:, cand_idx]
    rad_cand = rad_updated[:, cand_idx]
    dep_cand = dep_updated[:, cand_idx]
    con_cand = con_updated[:, cand_idx]
    col_cand = perturbed["colors"][None, cand_idx]
    opac_cand = perturbed["opacities"][None, cand_idx]

    _, isect_ids_cand, flat_ids_cand = gsplat.isect_tiles(
        m2d_cand, rad_cand, dep_cand, tile_size, tile_w, tile_h
    )
    offsets_cand = gsplat.isect_offset_encode(isect_ids_cand, 1, tile_w, tile_h)
    
    # Step 6: Sparse rasterization
    sparse_cand, _ = gsplat.rasterize_to_pixels(
        m2d_cand, con_cand, col_cand, opac_cand,
        width, height, tile_size, offsets_cand, flat_ids_cand,
        masks=invalid_mask[None, ...]
    )
    
    # Step 7: Frame compositing
    pixel_mask = invalid_mask.repeat_interleave(tile_size, dim=0).repeat_interleave(tile_size, dim=1)[:height, :width, None]
    composited = composite_with_cache(cached_frame, sparse_cand[0], pixel_mask)
    return composited

# Warmup
for _ in range(15):
    _ = run_partitioned_pipeline()
torch.cuda.synchronize()

starts_part = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
ends_part = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
for i in range(reps):
    starts_part[i].record()
    _ = run_partitioned_pipeline()
    ends_part[i].record()
torch.cuda.synchronize()
times_part = [starts_part[i].elapsed_time(ends_part[i]) for i in range(reps)]
med_part = np.median(times_part)
std_part = np.std(times_part)
print(f"Milestone 2B Partitioned Pipeline: {med_part:.4f} ms ± {std_part:.4f} ms")
print(f"End-to-End Speedup Ratio: {med_full / med_part:.2f}x")
