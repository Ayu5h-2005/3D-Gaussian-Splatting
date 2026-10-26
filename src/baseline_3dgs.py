"""
baseline_3dgs.py - Procedural 3D Gaussian Splatting Baseline Renderer

Renders a procedural scene containing 5,000 3D Gaussians using PyTorch and
the gsplat.rasterization API on an NVIDIA GPU (RTX 4060).

Outputs:
  - Rendered image: D:\3dgs-incremental-research\results\baseline_3dgs_render.png
  - Reusable scene parameters: D:\3dgs-incremental-research\results\procedural_scene_5k.pt
"""

import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------
# 1. Dependency & CUDA Verification
# -----------------------------------------------------------------------------
def verify_environment() -> None:
    """Verifies that PyTorch and gsplat are installed and CUDA is available."""
    try:
        import torch
    except ImportError as e:
        sys.exit(f"[ERROR] PyTorch is not installed in this environment: {e}")

    if not torch.cuda.is_available():
        sys.exit(
            "[ERROR] CUDA is not available to PyTorch. A CUDA-capable GPU is strictly "
            "required for 3D Gaussian Splatting rasterization."
        )

    try:
        import gsplat
    except ImportError as e:
        sys.exit(f"[ERROR] gsplat is not installed in this environment: {e}")

    # Verify gsplat CUDA extension availability
    try:
        from gsplat.cuda._wrapper import _make_lazy_cuda_obj
        # Testing if compiled C++/CUDA extension is initialized
        from gsplat import cuda
    except Exception as e:
        sys.exit(
            f"[ERROR] Failed to import gsplat CUDA wrapper: {e}\n"
            "Ensure gsplat was installed with precompiled CUDA binaries."
        )


# -----------------------------------------------------------------------------
# 2. Procedural Scene Generation
# -----------------------------------------------------------------------------
def generate_procedural_scene(
    num_gaussians: int = 5000,
    device: "torch.device" = None,
    seed: int = 42,
) -> Dict[str, "torch.Tensor"]:
    """
    Generates a deterministic 3D Gaussian scene in front of a camera.

    The Gaussians form a colorful 3D geometric structure (concentric spheres
    with spatial color gradients) positioned along the +Z axis in front of the camera.
    """
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Spherical coordinate distribution for spatial structure
    # Half outer shell, half inner core
    n_outer = num_gaussians // 2
    n_inner = num_gaussians - n_outer

    # Outer shell: radius ~ 0.8
    phi_out = torch.rand(n_outer, device=device) * (2.0 * np.pi)
    costheta_out = torch.rand(n_outer, device=device) * 2.0 - 1.0
    theta_out = torch.acos(costheta_out)
    r_out = 0.75 + torch.randn(n_outer, device=device) * 0.08

    x_out = r_out * torch.sin(theta_out) * torch.cos(phi_out)
    y_out = r_out * torch.sin(theta_out) * torch.sin(phi_out)
    z_out = r_out * torch.cos(theta_out)

    # Inner core: radius ~ 0.35
    phi_in = torch.rand(n_inner, device=device) * (2.0 * np.pi)
    costheta_in = torch.rand(n_inner, device=device) * 2.0 - 1.0
    theta_in = torch.acos(costheta_in)
    r_in = 0.35 + torch.randn(n_inner, device=device) * 0.05

    x_in = r_in * torch.sin(theta_in) * torch.cos(phi_in)
    y_in = r_in * torch.sin(theta_in) * torch.sin(phi_in)
    z_in = r_in * torch.cos(theta_in)

    x = torch.cat([x_out, x_in], dim=0)
    y = torch.cat([y_out, y_in], dim=0)
    z = torch.cat([z_out, z_in], dim=0)

    # Shift along +Z so all Gaussians are strictly in front of the camera (Z in [1.5, 3.5])
    z_center = 2.5
    means = torch.stack([x, y, z + z_center], dim=-1).to(torch.float32)

    # 1. Scales: strictly positive anisotropic standard deviations
    # Outer Gaussians slightly larger, inner denser
    scales_out = torch.rand((n_outer, 3), device=device) * 0.025 + 0.015
    scales_in = torch.rand((n_inner, 3), device=device) * 0.015 + 0.008
    scales = torch.cat([scales_out, scales_in], dim=0).to(torch.float32)

    # 2. Quaternions: random rotations normalized to unit length [w, x, y, z]
    quats = torch.randn((num_gaussians, 4), device=device, dtype=torch.float32)
    quats = torch.nn.functional.normalize(quats, p=2, dim=-1)

    # 3. Opacities: values in [0.2, 0.95]
    opacities = torch.rand(num_gaussians, device=device, dtype=torch.float32) * 0.75 + 0.20

    # 4. Colors: direct RGB values in [0.0, 1.0] with vibrant spatial mapping
    # Outer sphere: cyan/blue/magenta palette; Inner sphere: gold/orange/yellow
    norm_pos_out = (torch.stack([x_out, y_out, z_out], dim=-1) / 0.85).clamp(-1.0, 1.0)
    colors_out = torch.stack([
        0.5 + 0.45 * norm_pos_out[:, 0],
        0.4 + 0.50 * norm_pos_out[:, 1],
        0.8 + 0.18 * norm_pos_out[:, 2],
    ], dim=-1).clamp(0.0, 1.0)

    colors_in = torch.stack([
        torch.ones(n_inner, device=device) * 0.95,
        torch.rand(n_inner, device=device) * 0.4 + 0.5,
        torch.rand(n_inner, device=device) * 0.2 + 0.05,
    ], dim=-1).clamp(0.0, 1.0)

    colors = torch.cat([colors_out, colors_in], dim=0).to(torch.float32)

    return {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "colors": colors,
    }


# -----------------------------------------------------------------------------
# 3. Baseline Renderer Execution
# -----------------------------------------------------------------------------
def run_baseline_render(
    num_gaussians: int = 5000,
    width: int = 512,
    height: int = 512,
    output_dir: Path = Path(r"D:\3dgs-incremental-research\results"),
) -> None:
    """Executes the baseline 3DGS render using gsplat.rasterization."""
    import torch
    import gsplat

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)

    print("==================================================================")
    print("3D Gaussian Splatting Baseline Renderer")
    print("==================================================================")
    print(f"Device:           {gpu_name}")
    print(f"PyTorch Version:  {torch.__version__}")
    print(f"gsplat Version:   {gsplat.__version__}")
    print(f"Gaussian Count:   {num_gaussians:,}")
    print(f"Resolution:       {width} x {height}")
    print("------------------------------------------------------------------")

    # Generate procedural 3D Gaussians
    print("[1/4] Generating procedural 3D Gaussian parameters...")
    scene = generate_procedural_scene(num_gaussians=num_gaussians, device=device, seed=42)

    # Setup virtual camera (OpenCV coordinate frame: +X right, +Y down, +Z forward)
    # Camera at origin looking directly down +Z at the scene
    print("[2/4] Setting up camera matrices (512x512 pinhole)...")
    viewmats = torch.eye(4, device=device, dtype=torch.float32)[None, :, :]  # [1, 4, 4]

    fx = float(width)
    fy = float(height)
    cx = float(width) / 2.0
    cy = float(height) / 2.0
    Ks = torch.tensor([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0],
    ], device=device, dtype=torch.float32)[None, :, :]  # [1, 3, 3]

    # Save scene parameters for reproducible future benchmarking
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_file = output_dir / "procedural_scene_5k.pt"
    print(f"[3/4] Saving scene parameters to: {scene_file}")
    torch.save(
        {
            "means": scene["means"].detach().cpu(),
            "scales": scene["scales"].detach().cpu(),
            "quats": scene["quats"].detach().cpu(),
            "opacities": scene["opacities"].detach().cpu(),
            "colors": scene["colors"].detach().cpu(),
            "viewmats": viewmats.detach().cpu(),
            "Ks": Ks.detach().cpu(),
            "width": width,
            "height": height,
        },
        str(scene_file),
    )

    # Execute gsplat.rasterization with CUDA timing events
    print("[4/4] Executing gsplat.rasterization...")

    # Warmup pass to ensure CUDA context and kernel caches are primed
    try:
        _ = gsplat.rasterization(
            means=scene["means"],
            quats=scene["quats"],
            scales=scene["scales"],
            opacities=scene["opacities"],
            colors=scene["colors"],
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            render_mode="RGB",
        )
        torch.cuda.synchronize()
    except Exception as e:
        sys.exit(
            f"\n[FATAL ERROR] gsplat.rasterization failed during CUDA execution:\n{e}\n\n"
            "Diagnostic detail: The gsplat Python package requires a compiled CUDA extension "
            "matching your PyTorch and CUDA runtime."
        )

    # Timed benchmark pass
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    render_colors, render_alphas, meta = gsplat.rasterization(
        means=scene["means"],
        quats=scene["quats"],
        scales=scene["scales"],
        opacities=scene["opacities"],
        colors=scene["colors"],
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        render_mode="RGB",
    )
    end_event.record()
    torch.cuda.synchronize()

    render_time_ms = start_event.elapsed_time(end_event)

    # Validate output tensor shapes
    # render_colors: [1, height, width, 3] in [0.0, 1.0]
    assert render_colors.shape == (1, height, width, 3), (
        f"Unexpected render_colors shape: {render_colors.shape}"
    )

    # Convert to 8-bit RGB NumPy image and save via PIL
    rgb_tensor = render_colors[0].clamp(0.0, 1.0)
    rgb_numpy = (rgb_tensor.detach().cpu().numpy() * 255.0).astype(np.uint8)

    img_out_path = output_dir / "baseline_3dgs_render.png"
    Image.fromarray(rgb_numpy, mode="RGB").save(str(img_out_path))

    # Calculate basic rendering stats
    alpha_coverage = (render_alphas[0, :, :, 0] > 0.05).float().mean().item() * 100.0

    print("==================================================================")
    print("Baseline Render Complete & Verified")
    print("==================================================================")
    print(f"GPU Used:              {gpu_name}")
    print(f"Gaussian Count:        {num_gaussians:,}")
    print(f"Render Resolution:     {width} x {height}")
    print(f"Kernel Render Time:    {render_time_ms:.3f} ms ({1000.0 / render_time_ms:.1f} FPS)")
    print(f"Alpha Screen Coverage: {alpha_coverage:.2f}%")
    print(f"Saved Image:           {img_out_path}")
    print(f"Saved Scene Data:      {scene_file}")
    print("==================================================================")


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    verify_environment()
    run_baseline_render(
        num_gaussians=5000,
        width=512,
        height=512,
        output_dir=Path(r"D:\3dgs-incremental-research\results"),
    )
