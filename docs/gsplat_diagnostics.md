# gsplat Environment Diagnostics & Baseline Verification Report

**Project:** Real-Time Dynamic Scene Updating in 3D Gaussian Splatting Using Change Detection and Incremental Rasterization  
**Workspace:** `D:\3dgs-incremental-research`  
**Target Environment:** `D:\miniconda3\envs\gsplat310\python.exe`  
**Execution Date:** 2026-10-22  
**Status:** Baseline GPU Renderer Verified & Operational

---

## 1. Environment Diagnostic Results

A diagnostic probe was executed on `D:\miniconda3\envs\gsplat310\python.exe` to evaluate GPU availability, PyTorch CUDA bindings, and the `gsplat` C++/CUDA backend:

| Metric / Parameter | Diagnostic Finding | Verification Status |
| :--- | :--- | :--- |
| **Python Executable** | `D:\miniconda3\envs\gsplat310\python.exe` | Verified |
| **Python Version** | `3.10.22` (64-bit AMD64, MSC v.1942) | Verified |
| **PyTorch Version** | `2.4.1+cu121` | Verified |
| **PyTorch CUDA Available** | `True` | Verified |
| **torch.version.cuda** | `12.1` | Verified |
| **GPU Model** | NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB GDDR6 VRAM) | Verified |
| **CUDA Compute Capability**| `(8, 9)` (Ada Lovelace architecture) | Verified |
| **gsplat Package Version** | `1.5.3+pt24cu124` | Verified |
| **gsplat Location** | `D:\miniconda3\envs\gsplat310\lib\site-packages\gsplat` | Verified |
| **CUDA Backend (`_C`)** | `<module 'gsplat.csrc' from '...\gsplat\csrc.pyd'>` | **LOADED & FUNCTIONAL** |
| **Binary Extension Size** | `70,926,848 bytes` (~70.9 MB) | Verified |

---

## 2. Root Cause Analysis & Compatibility Assessment

### Why `3dgs_research` (Python 3.11) Failed Previously:
In the earlier `3dgs_research` environment, `pip install gsplat` pulled the pure-Python source distribution from PyPI. That distribution relies on runtime JIT compilation via `gsplat.cuda._backend._jit_compile()`, which requires an active `nvcc` compiler from the NVIDIA CUDA Toolkit. Because no local CUDA Toolkit was installed on the system, `_find_cuda_home()` returned `None`, disabling the CUDA extension (`_C = None`) and raising:
```text
AttributeError: 'NoneType' object has no attribute 'CameraModelType'
```

### Why `gsplat310` (Python 3.10) Succeeded:
The `gsplat310` environment was configured with the precompiled binary wheel:
- **Wheel Filename:** `gsplat-1.5.3+pt24cu124-cp310-cp310-win_amd64.whl`
- **Wheel Metadata (`WHEEL`):** `Root-Is-Purelib: false`, `Tag: cp310-cp310-win_amd64`
- **Binary Extension:** Contains pre-built `gsplat/csrc.pyd` compiled directly for Windows x64.
- **CUDA Runtime Linking:** The precompiled `.pyd` dynamically links against PyTorch's bundled CUDA runtime libraries (`cudart64_*.dll`, `cublas64_*.dll`). Local `nvcc` is completely bypassed.
- **CUDA 12.1 vs 12.4 Compatibility:** Even though PyTorch is built with `cu121` and the wheel tag indicates `pt24cu124`, both operate seamlessly on CUDA 12.x application binary interfaces (ABI) under NVIDIA Driver 610.91 (which supports up to CUDA 13.x APIs).

---

## 3. Preservation of Existing Workspace Assets

All prior project assets remain intact and unmodified:
- **Python 3.11 Environment (`D:\miniconda3\envs\3dgs_research`):** Unchanged.
- **Stage 1 Prototype ([`src/change_detection.py`](file:///D:/3dgs-incremental-research/src/change_detection.py)):** Preserved.
- **Streamlit Dashboard ([`ui/app.py`](file:///D:/3dgs-incremental-research/ui/app.py)):** Preserved.
- **Synthetic Test Results (`results/before.png`, `results/after.png`, etc.):** Preserved.

---

## 4. Baseline 3DGS Execution & Verification

With `gsplat.csrc` verified, [`src/baseline_3dgs.py`](file:///D:/3dgs-incremental-research/src/baseline_3dgs.py) was executed using the `gsplat310` interpreter.

### Terminal Output from Baseline Run:
```text
==================================================================
3D Gaussian Splatting Baseline Renderer
==================================================================
Device:           NVIDIA GeForce RTX 4060 Laptop GPU
PyTorch Version:  2.4.1+cu121
gsplat Version:   1.5.3+pt24cu124
Gaussian Count:   5,000
Resolution:       512 x 512
------------------------------------------------------------------
[1/4] Generating procedural 3D Gaussian parameters...
[2/4] Setting up camera matrices (512x512 pinhole)...
[3/4] Saving scene parameters to: D:\3dgs-incremental-research\results\procedural_scene_5k.pt
[4/4] Executing gsplat.rasterization...
==================================================================
Baseline Render Complete & Verified
==================================================================
GPU Used:              NVIDIA GeForce RTX 4060 Laptop GPU
Gaussian Count:        5,000
Render Resolution:     512 x 512
Kernel Render Time:    3.480 ms (287.4 FPS)
Alpha Screen Coverage: 52.22%
Saved Image:           D:\3dgs-incremental-research\results\baseline_3dgs_render.png
Saved Scene Data:      D:\3dgs-incremental-research\results\procedural_scene_5k.pt
==================================================================
```

### Verification of Generated Outputs:

An automated inspection verified the validity and integrity of the rendered image and saved scene parameters:

```python
# Output Verification
Image exists: True  (D:\3dgs-incremental-research\results\baseline_3dgs_render.png)
Image Format: PNG | Mode: RGB | Resolution: 512 x 512 | Channels: 3
Pixel Value Range: Min [0, 0, 0] to Max [239, 217, 195]
Mean RGB Channels: [57.84, 45.79, 71.92] (Confirms non-trivial, vibrant geometric rendering)

Scene Data exists: True (D:\3dgs-incremental-research\results\procedural_scene_5k.pt)
Scene Tensors:
  - means:      torch.Size([5000, 3])  float32
  - scales:     torch.Size([5000, 3])  float32
  - quats:      torch.Size([5000, 4])  float32 (normalized unit quaternions)
  - opacities:  torch.Size([5000])     float32
  - colors:     torch.Size([5000, 3])  float32
  - viewmats:   torch.Size([1, 4, 4])  float32
  - Ks:         torch.Size([1, 3, 3])  float32
```

---

## 5. Next Steps: Advancing to Incremental Rasterization Research

Now that the genuine GPU-accelerated 3DGS rendering baseline is fully verified and benchmarked (~3.48 ms per full frame on RTX 4060), the project can move forward to Stage 2:

1. **Controlled Scene Perturbation Engine (`src/dynamic/scene_perturber.py`):**
   - Apply deterministic modifications to a known subset of Gaussians (e.g. translate 5% of Gaussians by $\Delta x$ or toggle opacities).
2. **Screen-Space Tile Invalidation (`src/incremental/tile_manager.py`):**
   - Project modified 3D Gaussians to screen-space bounding boxes and mark affected $16 \times 16$ tiles.
3. **Incremental Tile Dispatch:**
   - Investigate `gsplat.isect_offset_encode` and `gsplat.rasterize_to_pixels` to evaluate work reuse in clean tiles while recomputing dirty tiles.
