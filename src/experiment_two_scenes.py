"""
experiment_two_scenes.py - CLI Backend for Comparing Two 3DGS Scenes

Executes in D:\miniconda3\envs\gsplat310 (PyTorch, CUDA, gsplat 1.5.3).
Accepts two PLY or PT files, performs automatic change detection, screen-space invalidation,
incremental rasterization, and parity evaluation, then outputs images and JSON diagnostics.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Union

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from src.ply_io import load_3dgs_ply
from src.scene_comparison import compare_and_render_two_scenes


def _load_any_scene(path: Union[str, Path], device: torch.device) -> Dict[str, torch.Tensor]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Scene file does not exist: {p}")

    suffix = p.suffix.lower()
    if suffix == ".ply":
        return load_3dgs_ply(p, device=device)
    elif suffix in (".pt", ".pth", ".bin"):
        data = torch.load(str(p), weights_only=False)
        scene = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                scene[k] = v.to(device)
            else:
                scene[k] = v
        return scene
    else:
        # Try PLY first, then PyTorch
        try:
            return load_3dgs_ply(p, device=device)
        except Exception:
            return torch.load(str(p), weights_only=False)


def main():
    parser = argparse.ArgumentParser(description="Compare Two 3DGS Scenes CLI")
    parser.add_argument("--scene-a", type=str, required=True, help="Path to baseline scene PLY/PT")
    parser.add_argument("--scene-b", type=str, required=True, help="Path to updated scene PLY/PT")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "results" / "two_scenes"))
    parser.add_argument("--pos-tol", type=float, default=0.03, help="Spatial distance tolerance for matching")
    parser.add_argument("--feat-tol", type=float, default=0.08, help="Appearance/scale tolerance for matching")
    parser.add_argument("--disp-tol", type=float, default=0.35, help="Maximum displacement tolerance for motion matching")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--cam-x", type=float, default=0.0)
    parser.add_argument("--cam-y", type=float, default=0.0)
    parser.add_argument("--cam-z", type=float, default=0.0)
    parser.add_argument("--look-z", type=float, default=2.5)
    parser.add_argument("--focal-len", type=float, default=512.0)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--reps", type=int, default=10)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        err_res = {"success": False, "error": "CUDA GPU required for 3D Gaussian Splatting rasterization."}
        print("__JSON_START__")
        print(json.dumps(err_res))
        print("__JSON_END__", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda:0")

    try:
        # 1. Load both scenes
        scene_a = _load_any_scene(args.scene_a, device)
        scene_b = _load_any_scene(args.scene_b, device)

        # 2. Camera Setup
        viewmats = torch.eye(4, device=device, dtype=torch.float32)[None, :, :]
        # If camera position is offset, adjust viewmat translation
        viewmats[0, 0, 3] = -args.cam_x
        viewmats[0, 1, 3] = -args.cam_y
        viewmats[0, 2, 3] = -args.cam_z

        fx = args.focal_len
        fy = args.focal_len
        cx = args.width / 2.0
        cy = args.height / 2.0
        Ks = torch.tensor([
            [fx,  0.0, cx],
            [0.0, fy,  cy],
            [0.0, 0.0, 1.0],
        ], device=device, dtype=torch.float32)[None, :, :]

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 3. Execute Comparison
        res = compare_and_render_two_scenes(
            scene_a=scene_a,
            scene_b=scene_b,
            viewmats=viewmats,
            Ks=Ks,
            width=args.width,
            height=args.height,
            tile_size=args.tile_size,
            pos_tol=args.pos_tol,
            feat_tol=args.feat_tol,
            disp_tol=args.disp_tol,
            output_dir=out_dir,
            num_warmups=args.warmups,
            num_reps=args.reps,
        )

        res_json = json.dumps(res, indent=2)
        print("__JSON_START__")
        print(res_json)
        print("__JSON_END__")

    except Exception as e:
        err_res = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }
        print("__JSON_START__")
        print(json.dumps(err_res))
        print("__JSON_END__", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
