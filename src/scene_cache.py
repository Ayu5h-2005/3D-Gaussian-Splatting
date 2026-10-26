"""
scene_cache.py - Static Scene Cache & Validity Manager for 3DGS (Milestone 2B)

Manages persistent GPU cache for static Gaussians:
- Caches baseline rendered frame t0.
- Caches 2D projected means, conics, depths, and radii for static Gaussians.
- Validates cache freshness against camera pose (viewmats), intrinsics (Ks), resolution, and tile size.
- Provides fast in-place buffer updates for dynamic Gaussians without full-scene re-projection.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import gsplat


@torch.jit.script
def _fused_tile_invalidation_kernel(
    m2d_all: torch.Tensor,
    rad_all: torch.Tensor,
    dep_all: torch.Tensor,
    y_grid: torch.Tensor,
    x_grid: torch.Tensor,
    width: int,
    height: int,
    tile_size: int,
    tile_w: int,
    tile_h: int,
) -> torch.Tensor:
    valid = (dep_all > 0.01) & (rad_all[:, 0] > 0) & (rad_all[:, 1] > 0)
    xy_min = m2d_all - rad_all
    xy_max = m2d_all + rad_all
    in_screen = (xy_max[:, 0] >= 0) & (xy_max[:, 1] >= 0) & (xy_min[:, 0] < width) & (xy_min[:, 1] < height)
    active = (valid & in_screen).view(1, -1).float()

    uv_min_x = torch.clamp((xy_min[:, 0] / tile_size).floor().long(), 0, tile_w - 1).view(-1, 1)
    uv_max_x = torch.clamp((xy_max[:, 0] / tile_size).floor().long(), 0, tile_w - 1).view(-1, 1)
    uv_min_y = torch.clamp((xy_min[:, 1] / tile_size).floor().long(), 0, tile_h - 1).view(1, -1)
    uv_max_y = torch.clamp((xy_max[:, 1] / tile_size).floor().long(), 0, tile_h - 1).view(1, -1)

    Y = ((y_grid >= uv_min_y) & (y_grid <= uv_max_y)).float() * active
    X = ((x_grid >= uv_min_x) & (x_grid <= uv_max_x)).float()

    return torch.mm(Y, X) > 0


class StaticSceneCache:
    """
    Persistent GPU cache for static Gaussian Splatting rendering data.
    """

    def __init__(
        self,
        scene: Dict[str, Union[torch.Tensor, int]],
        tile_size: int = 16,
    ):
        self.tile_size = tile_size
        self.device = scene["means"].device
        self.width = int(scene["width"])
        self.height = int(scene["height"])
        self.tile_w = (self.width + tile_size - 1) // tile_size
        self.tile_h = (self.height + tile_size - 1) // tile_size
        self.total_tiles = self.tile_w * self.tile_h
        self.num_total_gaussians = scene["means"].shape[0]

        # Precompute coordinate grids for invalidation
        self.y_grid = torch.arange(self.tile_h, device=self.device).view(self.tile_h, 1)
        self.x_grid = torch.arange(self.tile_w, device=self.device).view(1, self.tile_w)

        # Initialize cache
        self.viewmats_cached: Optional[torch.Tensor] = None
        self.Ks_cached: Optional[torch.Tensor] = None
        self.cached_frame: Optional[torch.Tensor] = None

        self.cached_radii: Optional[torch.Tensor] = None
        self.cached_means2d: Optional[torch.Tensor] = None
        self.cached_depths: Optional[torch.Tensor] = None
        self.cached_conics: Optional[torch.Tensor] = None

        # Working buffers for in-place updates (avoids memory allocation during rendering)
        self.buf_radii: Optional[torch.Tensor] = None
        self.buf_means2d: Optional[torch.Tensor] = None
        self.buf_depths: Optional[torch.Tensor] = None
        self.buf_conics: Optional[torch.Tensor] = None

        self.initialize(scene)

    def initialize(self, scene: Dict[str, Union[torch.Tensor, int]]) -> None:
        """Computes baseline full render and pre-projects all Gaussians to initialize cache."""
        self.viewmats_cached = scene["viewmats"].clone()
        self.Ks_cached = scene["Ks"].clone()

        # 1. Baseline Full Render at t0
        colors_t0, _, _ = gsplat.rasterization(
            scene["means"], scene["quats"], scene["scales"],
            scene["opacities"], scene["colors"],
            scene["viewmats"], scene["Ks"], self.width, self.height
        )
        self.cached_frame = colors_t0[0].clone()

        # 2. Pre-project all Gaussians at baseline t0
        radii, means2d, depths, conics, _ = gsplat.fully_fused_projection(
            scene["means"], None, scene["quats"], scene["scales"],
            scene["viewmats"], scene["Ks"], self.width, self.height, packed=False
        )

        self.cached_radii = radii.clone()
        self.cached_means2d = means2d.clone()
        self.cached_depths = depths.clone()
        self.cached_conics = conics.clone()

        # Initialize working buffers
        self.buf_radii = radii.clone()
        self.buf_means2d = means2d.clone()
        self.buf_depths = depths.clone()
        self.buf_conics = conics.clone()

    def is_valid(
        self,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        width: int,
        height: int,
        tile_size: int,
    ) -> bool:
        """
        Verifies if the cache is valid for the current camera, resolution, and rasterization config.
        If camera moves or resolution changes, cache is invalid.
        """
        if self.viewmats_cached is None or self.Ks_cached is None or self.cached_frame is None:
            return False

        if width != self.width or height != self.height or tile_size != self.tile_size:
            return False

        if not torch.equal(viewmats, self.viewmats_cached):
            return False

        if not torch.equal(Ks, self.Ks_cached):
            return False

        return True

    def project_dynamic_subset(
        self,
        scene_dynamic: Dict[str, Union[torch.Tensor, int]],
        dynamic_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Projects ONLY the dynamic Gaussians using gsplat.fully_fused_projection.
        Leaves static Gaussians untouched in cache.
        """
        if dynamic_indices.numel() == 0:
            return (
                torch.empty((1, 0, 2), device=self.device, dtype=self.cached_radii.dtype),
                torch.empty((1, 0, 2), device=self.device, dtype=self.cached_means2d.dtype),
                torch.empty((1, 0), device=self.device, dtype=self.cached_depths.dtype),
                torch.empty((1, 0, 3), device=self.device, dtype=self.cached_conics.dtype),
            )

        rad_dyn, m2d_dyn, dep_dyn, con_dyn, _ = gsplat.fully_fused_projection(
            scene_dynamic["means"][dynamic_indices], None,
            scene_dynamic["quats"][dynamic_indices], scene_dynamic["scales"][dynamic_indices],
            scene_dynamic["viewmats"], scene_dynamic["Ks"],
            self.width, self.height, packed=False
        )
        return rad_dyn, m2d_dyn, dep_dyn, con_dyn

    def compute_invalidation_from_cache(
        self,
        dynamic_indices: torch.Tensor,
        rad_dyn: torch.Tensor,
        m2d_dyn: torch.Tensor,
        dep_dyn: torch.Tensor,
        generate_pixel_mask: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Computes conservative tile invalidation mask using cached old 2D projections (t-1)
        and freshly projected new 2D projections (t). Zero redundant projection calls.
        Uses JIT-fused kernel for fast GPU execution without host-device synchronization.
        """
        if dynamic_indices.numel() == 0:
            invalid_tile_mask = torch.zeros((self.tile_h, self.tile_w), dtype=torch.bool, device=self.device)
            invalid_pixel_mask = (
                torch.zeros((self.height, self.width, 1), dtype=torch.bool, device=self.device)
                if generate_pixel_mask else None
            )
            return invalid_tile_mask, invalid_pixel_mask

        # Old projected footprints from cache
        m2d_old = self.cached_means2d[0, dynamic_indices]
        rad_old = self.cached_radii[0, dynamic_indices]
        dep_old = self.cached_depths[0, dynamic_indices]

        # Unified 2K tensor
        m2d_all = torch.cat([m2d_old, m2d_dyn[0]], dim=0)
        rad_all = torch.cat([rad_old, rad_dyn[0]], dim=0)
        dep_all = torch.cat([dep_old, dep_dyn[0]], dim=0)

        invalid_tile_mask = _fused_tile_invalidation_kernel(
            m2d_all, rad_all, dep_all,
            self.y_grid, self.x_grid,
            self.width, self.height, self.tile_size, self.tile_w, self.tile_h
        )

        invalid_pixel_mask = None
        if generate_pixel_mask:
            invalid_pixel_mask = (
                invalid_tile_mask.repeat_interleave(self.tile_size, dim=0)
                .repeat_interleave(self.tile_size, dim=1)[:self.height, :self.width, None]
            )

        return invalid_tile_mask, invalid_pixel_mask

    def composite(
        self,
        sparse_render: torch.Tensor,
        invalid_tile_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Assembles final frame via 5D broadcast compositing directly with cached frame.
        Eliminates the 512x512x1 pixel mask allocation and repeat_interleave overhead.
        """
        if sparse_render.dim() == 4:
            sparse_render = sparse_render[0]

        mask_5d = invalid_tile_mask.view(self.tile_h, 1, self.tile_w, 1, 1)
        cached_5d = self.cached_frame.view(self.tile_h, self.tile_size, self.tile_w, self.tile_size, 3)
        sparse_5d = sparse_render.view(self.tile_h, self.tile_size, self.tile_w, self.tile_size, 3)
        out_5d = torch.where(mask_5d, sparse_5d, cached_5d)
        return out_5d.reshape(self.height, self.width, 3)

    def update_working_buffers(
        self,
        dynamic_indices: torch.Tensor,
        rad_dyn: torch.Tensor,
        m2d_dyn: torch.Tensor,
        dep_dyn: torch.Tensor,
        con_dyn: torch.Tensor,
    ) -> None:
        """
        Updates in-place working buffers with dynamic Gaussian 2D projections.
        Avoids allocating new 5,000-element tensors.
        """
        if dynamic_indices.numel() == 0:
            return

        self.buf_radii[0, dynamic_indices] = rad_dyn[0]
        self.buf_means2d[0, dynamic_indices] = m2d_dyn[0]
        self.buf_depths[0, dynamic_indices] = dep_dyn[0]
        self.buf_conics[0, dynamic_indices] = con_dyn[0]

    def reset_working_buffers_to_cache(self, dynamic_indices: torch.Tensor) -> None:
        """Restores modified entries in working buffers back to baseline cached values."""
        if dynamic_indices.numel() == 0:
            return

        self.buf_radii[0, dynamic_indices] = self.cached_radii[0, dynamic_indices]
        self.buf_means2d[0, dynamic_indices] = self.cached_means2d[0, dynamic_indices]
        self.buf_depths[0, dynamic_indices] = self.cached_depths[0, dynamic_indices]
        self.buf_conics[0, dynamic_indices] = self.cached_conics[0, dynamic_indices]

    def filter_candidate_gaussians(
        self,
        invalid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Filters Gaussians to only those whose 2D screen-space bounding box touches at least
        one invalidated (dirty) tile, using a GPU 2D summed-area table (integral image).
        
        Guarantees mathematical parity (PSNR = inf, SSIM = 1.0) with zero false negatives,
        as Gaussians not touching dirty tiles have zero contribution to any dirty tile pixels.
        
        Complexity: O(tile_h * tile_w) for prefix sum + O(N) GPU vectorized bounding-box lookups.
        """
        if not invalid_mask.any():
            return torch.empty((0,), dtype=torch.long, device=self.device)

        rad = self.buf_radii[0]
        m2d = self.buf_means2d[0]
        dep = self.buf_depths[0]

        valid = (dep > 0.01) & (rad[:, 0] > 0) & (rad[:, 1] > 0)
        u_min = torch.clamp(torch.floor((m2d[:, 0] - rad[:, 0]) / self.tile_size).long(), 0, self.tile_w - 1)
        u_max = torch.clamp(torch.floor((m2d[:, 0] + rad[:, 0]) / self.tile_size).long(), 0, self.tile_w - 1)
        v_min = torch.clamp(torch.floor((m2d[:, 1] - rad[:, 1]) / self.tile_size).long(), 0, self.tile_h - 1)
        v_max = torch.clamp(torch.floor((m2d[:, 1] + rad[:, 1]) / self.tile_size).long(), 0, self.tile_h - 1)

        # 2D Integral image of invalid_mask
        inv_int = invalid_mask.int()
        prefix = torch.zeros((self.tile_h + 1, self.tile_w + 1), dtype=torch.int32, device=self.device)
        prefix[1:, 1:] = torch.cumsum(torch.cumsum(inv_int, dim=0), dim=1)

        # Vectorized Bounding Box sum query on GPU
        box_dirty_count = (
            prefix[v_max + 1, u_max + 1]
            - prefix[v_min, u_max + 1]
            - prefix[v_max + 1, u_min]
            + prefix[v_min, u_min]
        )
        touches_dirty = (box_dirty_count > 0) & valid
        cand_indices = torch.nonzero(touches_dirty).squeeze(-1)
        return cand_indices

