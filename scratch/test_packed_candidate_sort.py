import sys
from pathlib import Path

ROOT = Path(r"D:\3dgs-incremental-research")
sys.path.insert(0, str(ROOT))

import torch
import gsplat
from src.scene_perturbation import perturb_scene
from src.tile_invalidation import compute_ground_truth_changed_tiles, compute_image_metrics

device = torch.device("cuda:0")
scene = torch.load(ROOT / "results" / "procedural_scene_5k.pt", weights_only=False)
for k, v in scene.items():
    if isinstance(v, torch.Tensor):
        scene[k] = v.to(device)

width, height = int(scene["width"]), int(scene["height"])
tile_size = 16
tile_w = width // tile_size
tile_h = height // tile_size

# Perturb 1% spatial cluster
perturbed, changed_idx, _ = perturb_scene(
    scene, fraction=0.01, delta=(0.15, -0.10, 0.05), seed=42, mode="spatial_cluster"
)

# Reference full rerender
ref_colors, _, _ = gsplat.rasterization(
    perturbed["means"], perturbed["quats"], perturbed["scales"],
    perturbed["opacities"], perturbed["colors"],
    perturbed["viewmats"], perturbed["Ks"], width, height
)
img_ref = ref_colors[0]

# Project perturbed scene
rad_p, m2d_p, dep_p, con_p, _ = gsplat.fully_fused_projection(
    perturbed["means"], None, perturbed["quats"], perturbed["scales"],
    perturbed["viewmats"], perturbed["Ks"], width, height, packed=False
)

# Pick a subset: say candidates touching dirty tiles or just the first 1000 Gaussians
cand_indices = torch.arange(0, 1000, device=device, dtype=torch.int64)

# Test packed isect_tiles with gaussian_ids!
nnz = cand_indices.numel()
m2d_sub = m2d_p[0, cand_indices].contiguous()
rad_sub = rad_p[0, cand_indices].contiguous()
dep_sub = dep_p[0, cand_indices].contiguous()
image_ids = torch.zeros(nnz, device=device, dtype=torch.int64)
gaussian_ids = cand_indices.contiguous()

tiles_per_gauss, isect_ids, flatten_ids = gsplat.isect_tiles(
    m2d_sub, rad_sub, dep_sub, tile_size, tile_w, tile_h,
    sort=True, packed=True, n_images=1, image_ids=image_ids, gaussian_ids=gaussian_ids
)

print("isect_tiles with packed=True succeeded!")
print("flatten_ids min/max:", flatten_ids.min().item(), flatten_ids.max().item())
print("flatten_ids dtype:", flatten_ids.dtype, "shape:", flatten_ids.shape)

offsets = gsplat.isect_offset_encode(isect_ids, 1, tile_w, tile_h)
print("offsets shape:", offsets.shape)

# Now test rasterize_to_pixels with FULL scene conics, colors, opacities using flatten_ids!
pert_colors = perturbed["colors"][None, ...]
pert_opacities = perturbed["opacities"][None, ...]

sparse_cols, _ = gsplat.rasterize_to_pixels(
    m2d_p, con_p, pert_colors, pert_opacities,
    width, height, tile_size, offsets, flatten_ids
)
print("rasterize_to_pixels with packed flatten_ids succeeded! Shape:", sparse_cols.shape)
