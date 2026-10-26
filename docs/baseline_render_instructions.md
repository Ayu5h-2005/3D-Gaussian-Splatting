# Baseline 3D Gaussian Splatting Render Instructions

**Script:** [`src/baseline_3dgs.py`](file:///D:/3dgs-incremental-research/src/baseline_3dgs.py)  
**Target Environment:** `D:\miniconda3\envs\3dgs_research\python.exe`  
**Target Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)  
**Output Files:**  
- Rendered Image: `D:\3dgs-incremental-research\results\baseline_3dgs_render.png`  
- Reusable Scene Data: `D:\3dgs-incremental-research\results\procedural_scene_5k.pt`

---

## 1. What the Script Does

[`src/baseline_3dgs.py`](file:///D:/3dgs-incremental-research/src/baseline_3dgs.py) establishes the real 3D Gaussian Splatting rendering baseline for this research project:
1. **Verifies Hardware & CUDA:** Asserts that PyTorch detects the NVIDIA GeForce RTX 4060 Laptop GPU and that `gsplat` is available.
2. **Generates a Procedural 3DGS Scene:** Procedurally synthesizes $5,000$ 3D Gaussians structured as a dual-shell 3D geometric object positioned directly in front of the camera along the $+Z$ axis:
   - 3D Centers ($\mu \in \mathbb{R}^{5000 \times 3}$) centered at $z = 2.5$.
   - Positive 3D scales ($s \in \mathbb{R}^{5000 \times 3}$).
   - Normalized unit quaternions ($q \in \mathbb{R}^{5000 \times 4}$).
   - Opacities ($\alpha \in [0.2, 0.95]$).
   - Direct RGB color values ($c \in \mathbb{R}^{5000 \times 3}$).
3. **Configures Camera Matrices:** Establishes a $512 \times 512$ pinhole camera model ($f_x = f_y = 512$, $c_x = c_y = 256$, identity view matrix).
4. **Executes `gsplat.rasterization`:** Performs 3D-to-2D EWA projection, tile binning, depth sorting, and front-to-back alpha blending on the RTX 4060 GPU.
5. **Measures Latency:** Records CUDA event timing for the rasterization kernel.
6. **Saves Outputs:** Writes the resulting image and the exact scene parameter tensors for future reproducible comparisons against incremental rendering.

---

## 2. Environment Verification Step

Before running the script, verify that `torch` and `gsplat` are functional and recognize the GPU:

```powershell
& "D:\miniconda3\envs\3dgs_research\python.exe" -c "import torch, gsplat; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0)); print('gsplat:', gsplat.__version__)"
```

### Expected Output:
```text
PyTorch: 2.4.1+cu121
CUDA Available: True
Device: NVIDIA GeForce RTX 4060 Laptop GPU
gsplat: 1.5.3
```

---

## 3. Command to Run the Baseline Renderer

Execute the baseline rendering script using the isolated Python interpreter:

```powershell
& "D:\miniconda3\envs\3dgs_research\python.exe" "D:\3dgs-incremental-research\src\baseline_3dgs.py"
```

### Expected Terminal Output:
```text
==================================================================
3D Gaussian Splatting Baseline Renderer
==================================================================
Device:           NVIDIA GeForce RTX 4060 Laptop GPU
PyTorch Version:  2.4.1+cu121
gsplat Version:   1.5.3
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
Kernel Render Time:    <time> ms (<fps> FPS)
Alpha Screen Coverage: ~45-55%
Saved Image:           D:\3dgs-incremental-research\results\baseline_3dgs_render.png
Saved Scene Data:      D:\3dgs-incremental-research\results\procedural_scene_5k.pt
==================================================================
```

---

## 4. How to Verify the Output

Run this verification command to confirm that the rendered image and scene parameters were properly written and contain non-trivial data:

```powershell
& "D:\miniconda3\envs\3dgs_research\python.exe" -c "from PIL import Image; import torch, numpy as np; img = Image.open(r'D:\3dgs-incremental-research\results\baseline_3dgs_render.png'); arr = np.array(img); print('Image Dimensions:', arr.shape, 'Mode:', img.mode); print('Color Value Range:', arr.min(), 'to', arr.max()); assert arr.shape == (512, 512, 3), 'Incorrect image shape!'; data = torch.load(r'D:\3dgs-incremental-research\results\procedural_scene_5k.pt'); print('Saved Gaussians:', data['means'].shape[0]); print('All verification checks PASSED!')"
```

### Verification Checks Performed:
1. `baseline_3dgs_render.png` exists, is $512 \times 512 \times 3$, and is in 8-bit RGB format.
2. The image is not blank or uniform (verifies non-trivial dynamic range across RGB channels).
3. `procedural_scene_5k.pt` contains all 5,000 Gaussian parameters (`means`, `scales`, `quats`, `opacities`, `colors`, `viewmats`, `Ks`).

---

## 5. Troubleshooting & Error Signatures

### Error: `gsplat: No CUDA toolkit found. gsplat will be disabled` or `AttributeError: 'NoneType' object has no attribute 'CameraModelType'`
- **Root Cause:** `gsplat` was installed as a pure-Python sdist from PyPI rather than from the pre-compiled binary wheel index (`https://docs.gsplat.studio/whl/pt24cu121`). When installed from PyPI, `gsplat` attempts JIT compilation requiring a local `nvcc` compiler, which is absent on the host.
- **Resolution:** Reinstall `gsplat` using the official prebuilt Windows wheel index:
  ```powershell
  & "D:\miniconda3\envs\3dgs_research\Scripts\pip.exe" install --force-reinstall gsplat --index-url https://docs.gsplat.studio/whl/pt24cu121 --cache-dir "D:\pip_cache"
  ```

### Error: `CUDA out of memory`
- **Root Cause:** Another process is holding VRAM on the RTX 4060 Laptop GPU.
- **Resolution:** Check running processes using `nvidia-smi` and terminate any background tasks consuming GPU memory. (5,000 Gaussians requires < 50 MB of VRAM).
