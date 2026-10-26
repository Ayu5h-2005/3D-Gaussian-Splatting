import sys
from pathlib import Path
ROOT = Path(r"D:\3dgs-incremental-research")
sys.path.insert(0, str(ROOT))

import torch
from src.baseline_3dgs import generate_procedural_scene
from src.ply_io import load_3dgs_ply

device = torch.device("cuda:0")
scene_a = load_3dgs_ply(ROOT / "results" / "ply_demo" / "baseline_scene.ply", device=device)
scene_b = load_3dgs_ply(ROOT / "results" / "ply_demo" / "updated_scene.ply", device=device)

def match_gaussians_two_stage(
    scene_a,
    scene_b,
    pos_tol: float = 0.03,
    feat_tol: float = 0.08,
    disp_tol: float = 0.35,
    chunk_size: int = 4000,
):
    pos_a = scene_a["means"]
    pos_b = scene_b["means"]
    na = pos_a.shape[0]
    nb = pos_b.shape[0]

    # --- STAGE 1: Co-located Matching (Static + In-Place Modifications) ---
    # Chunked mutual NN for all points
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

    is_mutual_s1 = (nearest_a_for_b[nearest_b_for_a] == torch.arange(na, device=device))
    pos_matched_s1 = is_mutual_s1 & (dist_b_for_a <= pos_tol)

    matched_b_s1 = nearest_b_for_a
    col_delta_a = torch.norm(scene_a["colors"] - scene_b["colors"][matched_b_s1], dim=-1)
    opac_delta_a = (scene_a["opacities"] - scene_b["opacities"][matched_b_s1]).abs()
    scale_delta_a = torch.norm(scene_a["scales"] - scene_b["scales"][matched_b_s1], dim=-1)
    feat_matched_s1 = (col_delta_a <= feat_tol) & (opac_delta_a <= feat_tol) & (scale_delta_a <= feat_tol)

    # Static: co-located and identical features
    static_a_mask = pos_matched_s1 & feat_matched_s1
    static_b_mask = torch.zeros(nb, dtype=torch.bool, device=device)
    static_b_mask[matched_b_s1[static_a_mask]] = True

    # In-place modified: co-located but features changed
    inplace_mod_a = pos_matched_s1 & (~feat_matched_s1)
    inplace_mod_b = torch.zeros(nb, dtype=torch.bool, device=device)
    if inplace_mod_a.any():
        inplace_mod_b[matched_b_s1[inplace_mod_a]] = True

    # --- STAGE 2: Motion / Displaced Matching on Residuals ---
    unmatched_a_idx = torch.where(~(static_a_mask | inplace_mod_a))[0]
    unmatched_b_idx = torch.where(~(static_b_mask | inplace_mod_b))[0]

    displaced_mod_a = torch.zeros(na, dtype=torch.bool, device=device)
    displaced_mod_b = torch.zeros(nb, dtype=torch.bool, device=device)

    if unmatched_a_idx.numel() > 0 and unmatched_b_idx.numel() > 0 and disp_tol > pos_tol:
        pos_un_a = pos_a[unmatched_a_idx]
        pos_un_b = pos_b[unmatched_b_idx]
        scales_un_a = scene_a["scales"][unmatched_a_idx]
        scales_un_b = scene_b[scales_un_a.shape[-1] and "scales"][unmatched_b_idx]
        opac_un_a = scene_a["opacities"][unmatched_a_idx]
        opac_un_b = scene_b["opacities"][unmatched_b_idx]
        quats_un_a = scene_a["quats"][unmatched_a_idx]
        quats_un_b = scene_b["quats"][unmatched_b_idx]

        # Geometry cost matrix on unmatched pool
        d_pos_un = torch.cdist(pos_un_a, pos_un_b)
        d_scale_un = torch.cdist(scales_un_a, scales_un_b)
        d_opac_un = (opac_un_a[:, None] - opac_un_b[None, :]).abs()
        dot_q = torch.abs(torch.mm(quats_un_a, quats_un_b.t()))
        d_quat = 1.0 - dot_q

        # Total invariant cost: geometric shape consistency
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

    # Total Modified
    modified_a_mask = inplace_mod_a | displaced_mod_a
    modified_b_mask = inplace_mod_b | displaced_mod_b

    # True Removals (in A, neither static nor modified)
    removed_a_mask = ~(static_a_mask | modified_a_mask)

    # True Additions (in B, neither static nor modified)
    added_b_mask = ~(static_b_mask | modified_b_mask)

    dirty_a_mask = ~static_a_mask
    dirty_b_mask = ~static_b_mask

    return {
        "num_static": int(static_a_mask.sum().item()),
        "num_modified": int(modified_a_mask.sum().item()),
        "num_removed": int(removed_a_mask.sum().item()),
        "num_added": int(added_b_mask.sum().item()),
        "static_a_mask": static_a_mask,
        "modified_a_mask": modified_a_mask,
        "removed_a_mask": removed_a_mask,
        "added_b_mask": added_b_mask,
        "dirty_a_mask": dirty_a_mask,
        "dirty_b_mask": dirty_b_mask,
    }

res = match_gaussians_two_stage(scene_a, scene_b)
print("Two-Stage Matching Results:")
print(f"  Static matched: {res['num_static']} (expected 4851)")
print(f"  Modified:       {res['num_modified']} (expected 49)")
print(f"  Removed:        {res['num_removed']} (expected 100)")
print(f"  Added:          {res['num_added']} (expected 150)")
