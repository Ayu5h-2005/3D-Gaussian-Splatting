import sys
from pathlib import Path

ROOT = Path(r"D:\3dgs-incremental-research")
sys.path.insert(0, str(ROOT))

import torch
import gsplat
from src.scene_perturbation import perturb_scene
from src.tile_invalidation import composite_with_cache

device = torch.device("cuda:0")
scene = torch.load(ROOT / "results" / "procedural_scene_5k.pt", weights_only=False)
for k, v in scene.items():
    if isinstance(v, torch.Tensor):
        scene[k] = v.to(device)

width, height = int(scene["width"]), int(scene["height"])
tile_size = 16
tile_w, tile_h = width // tile_size, height // tile_size

colors_t0, _, _ = gsplat.rasterization(
    scene["means"], scene["quats"], scene["scales"],
    scene["opacities"], scene["colors"],
    scene["viewmats"], scene["Ks"], width, height
)
cached_frame = colors_t0[0].clone()

rad_static, m2d_static, dep_static, con_static, _ = gsplat.fully_fused_projection(
    scene["means"], None, scene["quats"], scene["scales"],
    scene["viewmats"], scene["Ks"], width, height, packed=False
)

perturbed, changed_idx, _ = perturb_scene(
    scene, fraction=0.01, delta=(0.12, -0.10, 0.04), seed=42, mode="spatial_cluster"
)

ev = [torch.cuda.Event(enable_timing=True) for _ in range(10)]

for _ in range(10):
    ev[0].record()
    # 1. Dyn proj
    rad_dyn, m2d_dyn, dep_dyn, con_dyn, _ = gsplat.fully_fused_projection(
        perturbed["means"][changed_idx], None,
        perturbed["quats"][changed_idx], perturbed["scales"][changed_idx],
        perturbed["viewmats"], perturbed["Ks"], width, height, packed=False
    )
    ev[1].record()
    # 2. Invalidation
    m2d_all_c = torch.cat([m2d_static[0, changed_idx], m2d_dyn[0]], dim=0)
    rad_all_c = torch.cat([rad_static[0, changed_idx], rad_dyn[0]], dim=0)
    dep_all_c = torch.cat([dep_static[0, changed_idx], dep_dyn[0]], dim=0)
    valid_c = (dep_all_c > 0.01) & (rad_all_c > 0).all(dim=-1)
    xy_min_c = m2d_all_c - rad_all_c
    xy_max_c = m2d_all_c + rad_all_c
    uv_min_c = torch.clamp((xy_min_c / tile_size).floor().long(), 0, tile_w - 1)
    uv_max_c = torch.clamp((xy_max_c / tile_size).floor().long(), 0, tile_h - 1)
    y_grid = torch.arange(tile_h, device=device).view(tile_h, 1)
    x_grid = torch.arange(tile_w, device=device).view(1, tile_w)
    Y = ((y_grid >= uv_min_c[:, 1].view(1, -1)) & (y_grid <= uv_max_c[:, 1].view(1, -1))).float() * valid_c.view(1, -1).float()
    X = ((x_grid >= uv_min_c[:, 0].view(-1, 1)) & (x_grid <= uv_max_c[:, 0].view(-1, 1))).float()
    invalid_mask = (torch.mm(Y, X) > 0)
    ev[2].record()
    # 3. Buffer update
    m2d_updated = m2d_static.clone()
    m2d_updated[0, changed_idx] = m2d_dyn[0]
    ev[3].record()
    # 4. Prefix sum & box query
    inv_int = invalid_mask.int()
    prefix = torch.zeros((tile_h + 1, tile_w + 1), dtype=torch.int32, device=device)
    prefix[1:, 1:] = torch.cumsum(torch.cumsum(inv_int, dim=0), dim=1)
    ev[4].record()
    # 5. Candidate selection
    rad_u = rad_static[0]
    m2d_u = m2d_updated[0]
    u_min_u = torch.clamp(torch.floor((m2d_u[:, 0] - rad_u[:, 0]) / tile_size).long(), 0, tile_w - 1)
    u_max_u = torch.clamp(torch.floor((m2d_u[:, 0] + rad_u[:, 0]) / tile_size).long(), 0, tile_w - 1)
    v_min_u = torch.clamp(torch.floor((m2d_u[:, 1] - rad_u[:, 1]) / tile_size).long(), 0, tile_h - 1)
    v_max_u = torch.clamp(torch.floor((m2d_u[:, 1] + rad_u[:, 1]) / tile_size).long(), 0, tile_h - 1)
    box_dirty = (
        prefix[v_max_u + 1, u_max_u + 1]
        - prefix[v_min_u, u_max_u + 1]
        - prefix[v_max_u + 1, u_min_u]
        + prefix[v_min_u, u_min_u]
    )
    touches_dirty = (box_dirty > 0)
    cand_idx = torch.nonzero(touches_dirty).squeeze(-1)
    ev[5].record()
    # 6. Slicing candidates
    m2d_cand = m2d_updated[:, cand_idx]
    rad_cand = rad_static[:, cand_idx]
    dep_cand = dep_static[:, cand_idx]
    con_cand = con_static[:, cand_idx]
    col_cand = perturbed["colors"][None, cand_idx]
    opac_cand = perturbed["opacities"][None, cand_idx]
    ev[6].record()
    # 7. isect_tiles
    _, isect_ids_cand, flat_ids_cand = gsplat.isect_tiles(
        m2d_cand, rad_cand, dep_cand, tile_size, tile_w, tile_h
    )
    offsets_cand = gsplat.isect_offset_encode(isect_ids_cand, 1, tile_w, tile_h)
    ev[7].record()
    # 8. rasterize
    sparse_cand, _ = gsplat.rasterize_to_pixels(
        m2d_cand, con_cand, col_cand, opac_cand,
        width, height, tile_size, offsets_cand, flat_ids_cand,
        masks=invalid_mask[None, ...]
    )
    ev[8].record()
    # 9. composite
    pixel_mask = invalid_mask.repeat_interleave(tile_size, dim=0).repeat_interleave(tile_size, dim=1)[:height, :width, None]
    comp = composite_with_cache(cached_frame, sparse_cand[0], pixel_mask)
    ev[9].record()

torch.cuda.synchronize()

print(f"1. Dyn Proj: {ev[0].elapsed_time(ev[1]):.3f} ms")
print(f"2. Invalidation: {ev[1].elapsed_time(ev[2]):.3f} ms")
print(f"3. Buffer update: {ev[2].elapsed_time(ev[3]):.3f} ms")
print(f"4. Prefix sum: {ev[3].elapsed_time(ev[4]):.3f} ms")
print(f"5. Candidate query: {ev[4].elapsed_time(ev[5]):.3f} ms")
print(f"6. Candidate slice: {ev[5].elapsed_time(ev[6]):.3f} ms")
print(f"7. isect_tiles on candidates: {ev[6].elapsed_time(ev[7]):.3f} ms")
print(f"8. rasterize on candidates: {ev[7].elapsed_time(ev[8]):.3f} ms")
print(f"9. composite: {ev[8].elapsed_time(ev[9]):.3f} ms")
print(f"TOTAL: {ev[0].elapsed_time(ev[9]):.3f} ms")
