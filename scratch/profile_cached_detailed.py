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

rad_cache, m2d_cache, dep_cache, con_cache, _ = gsplat.fully_fused_projection(
    scene["means"], None, scene["quats"], scene["scales"],
    scene["viewmats"], scene["Ks"], width, height, packed=False
)
m2d_buf = m2d_cache.clone()
rad_buf = rad_cache.clone()
dep_buf = dep_cache.clone()
con_buf = con_cache.clone()

y_grid = torch.arange(tile_h, device=device).view(tile_h, 1)
x_grid = torch.arange(tile_w, device=device).view(1, tile_w)

perturbed, changed_idx, _ = perturb_scene(
    scene, fraction=0.01, delta=(0.12, -0.10, 0.04), seed=42, mode="spatial_cluster"
)

def execute_pipeline(record_events=False):
    if record_events: e[0].record()
    rad_dyn, m2d_dyn, dep_dyn, con_dyn, _ = gsplat.fully_fused_projection(
        perturbed["means"][changed_idx], None,
        perturbed["quats"][changed_idx], perturbed["scales"][changed_idx],
        perturbed["viewmats"], perturbed["Ks"], width, height, packed=False
    )
    if record_events: e[1].record()

    m2d_o = m2d_cache[0, changed_idx]
    rad_o = rad_cache[0, changed_idx]
    dep_o = dep_cache[0, changed_idx]
    m2d_all = torch.cat([m2d_o, m2d_dyn[0]], dim=0)
    rad_all = torch.cat([rad_o, rad_dyn[0]], dim=0)
    dep_all = torch.cat([dep_o, dep_dyn[0]], dim=0)
    valid = (dep_all > 0.01) & (rad_all > 0).all(dim=-1)
    xy_min = m2d_all - rad_all
    xy_max = m2d_all + rad_all
    in_screen = (xy_max >= 0).all(dim=-1) & (xy_min[:, 0] < width) & (xy_min[:, 1] < height)
    active = valid & in_screen
    uv_min = torch.clamp((xy_min / tile_size).floor().long(), 0, tile_w - 1)
    uv_max = torch.clamp((xy_max / tile_size).floor().long(), 0, tile_h - 1)
    Y = ((y_grid >= uv_min[:, 1].view(1, -1)) & (y_grid <= uv_max[:, 1].view(1, -1))).float() * active.view(1, -1).float()
    X = ((x_grid >= uv_min[:, 0].view(-1, 1)) & (x_grid <= uv_max[:, 0].view(-1, 1))).float()
    invalid_mask = (torch.mm(Y, X) > 0)
    if record_events: e[2].record()

    m2d_buf[0, changed_idx] = m2d_dyn[0]
    rad_buf[0, changed_idx] = rad_dyn[0]
    dep_buf[0, changed_idx] = dep_dyn[0]
    con_buf[0, changed_idx] = con_dyn[0]
    if record_events: e[3].record()

    _, isect_ids, flatten_ids = gsplat.isect_tiles(
        m2d_buf, rad_buf, dep_buf, tile_size, tile_w, tile_h
    )
    if record_events: e[4].record()

    offsets = gsplat.isect_offset_encode(isect_ids, 1, tile_w, tile_h)
    if record_events: e[5].record()

    sparse_colors, _ = gsplat.rasterize_to_pixels(
        m2d_buf, con_buf, perturbed["colors"][None, ...], perturbed["opacities"][None, ...],
        width, height, tile_size, offsets, flatten_ids,
        masks=invalid_mask[None, ...]
    )
    if record_events: e[6].record()

    pixel_mask = invalid_mask.repeat_interleave(tile_size, dim=0).repeat_interleave(tile_size, dim=1)[:height, :width, None]
    comp = composite_with_cache(cached_frame, sparse_colors[0], pixel_mask)
    if record_events: e[7].record()
    return comp

# Dedicated Warmup of entire pipeline
for _ in range(15):
    execute_pipeline(record_events=False)
torch.cuda.synchronize()

e = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
execute_pipeline(record_events=True)
torch.cuda.synchronize()

torch.cuda.synchronize()

print(f"1. Dynamic Proj: {e[0].elapsed_time(e[1]):.4f} ms")
print(f"2. Invalidation: {e[1].elapsed_time(e[2]):.4f} ms")
print(f"3. In-place update: {e[2].elapsed_time(e[3]):.4f} ms")
print(f"4. isect_tiles: {e[3].elapsed_time(e[4]):.4f} ms")
print(f"5. isect_offset_encode: {e[4].elapsed_time(e[5]):.4f} ms")
print(f"6. rasterize_to_pixels: {e[5].elapsed_time(e[6]):.4f} ms")
print(f"7. compositing: {e[6].elapsed_time(e[7]):.4f} ms")
print(f"TOTAL: {e[0].elapsed_time(e[7]):.4f} ms")
