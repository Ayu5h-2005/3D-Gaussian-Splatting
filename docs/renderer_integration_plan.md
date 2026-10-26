# 3D Gaussian Splatting Renderer Integration Plan

**Project:** Real-Time Dynamic Scene Updating in 3D Gaussian Splatting Using Change Detection and Incremental Rasterization  
**Workspace:** `D:\3dgs-incremental-research`  
**Target Environment:** Windows 11 Home (x64) | NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)  
**Python Runtime:** `D:\miniconda3\envs\3dgs_research\python.exe` (Python 3.11.17)  
**Status:** Planning & Verification Phase (Baseline Renderer Integration)

---

## 1. Executive Summary & Research Context

The objective of this research project is to investigate whether rendering computations can be significantly reduced in dynamic 3D Gaussian Splatting (3DGS) scenes by detecting localized changes, identifying affected screen-space tiles, and re-rasterizing only the affected regions while reusing cached pixel data for unaffected tiles.

### Current Implementation State
- **Stage 1 Prototype (Synthetic):**
  - [`src/change_detection.py`](file:///D:/3dgs-incremental-research/src/change_detection.py) demonstrates 2D image differencing and maps changed pixels to $32\times 32$ screen tiles.
  - [`ui/app.py`](file:///D:/3dgs-incremental-research/ui/app.py) provides an interactive Streamlit dashboard visualizing tile reuse estimates on synthetic rectangles.
- **Limitation:** The current prototype operates entirely on synthetic 2D bitmaps. It does not evaluate 3D Gaussian projections, depth ordering, or alpha compositing.
- **Immediate Milestone:** Integrate a real, high-performance 3DGS renderer, verify it on the local RTX 4060 GPU, and establish a reproducible rendering baseline for a small scene without modifying rendering behavior yet.

> [!IMPORTANT]
> **Scientific Integrity Notice:** No incremental rendering performance claims or speedups are asserted at this stage. Real-time dynamic updating remains a research goal to be empirically measured against full-frame baseline re-rendering.

---

## 2. Environment & Hardware Audit

A comprehensive hardware and runtime inspection was conducted on the host machine.

### Hardware & Operating System
| Component | Verified Specification | Impact on 3DGS Integration |
| :--- | :--- | :--- |
| **OS** | Windows 11 Home (64-bit, Build 10.0.26340) | Native Windows environment. Requires Windows-compatible wheels or MSVC build toolchains. |
| **GPU** | NVIDIA GeForce RTX 4060 Laptop GPU | Ada Lovelace architecture (Compute Capability `sm_89`). Fully supports modern CUDA kernels. |
| **VRAM** | 8,188 MiB (~8.0 GB GDDR6) | Ample memory for small/medium 3DGS scenes (~100k–500k Gaussians) at 720p/1080p. |
| **Driver** | NVIDIA Display Driver 610.91 (WDDM mode) | CUDA Kernel Mode Driver supports up to CUDA 13.x APIs. |
| **Disk Storage** | **`C:\`:** 292.1 GB Total (**17.53 GB Free** - LOW)<br>**`D:\`:** 329.9 GB Total (**232.38 GB Free** - PLENTIFUL) | **Strict Rule:** All downloads, virtual environments, pip caches, and datasets must reside on `D:\` to protect `C:\`. |

### Python Environment Audit (`3dgs_research`)
- **Python Version:** `3.11.17` (64-bit AMD64) located at `D:\miniconda3\envs\3dgs_research\python.exe`.
- **System Python (`3.13.1`):** Untouched and isolated at `C:\Users\aptha\AppData\Local\Programs\Python\Python313`.
- **Installed Packages:** `numpy` (2.4.6), `opencv-python` (5.0.0.93), `pillow` (13.0.0), `matplotlib` (3.11.2), `pandas` (3.1.0), `streamlit` (1.66.1).
- **PyTorch Status:** **Not installed** (`No module named 'torch'`).
- **CUDA Toolkit (`nvcc`):** **Not installed / Not on system PATH**.

---

## 3. Comparative Assessment of 3DGS Renderers

We evaluated the primary existing 3D Gaussian Splatting implementations for technical feasibility, Windows compatibility, and suitability for tile-level incremental modifications.

### Comparison Matrix

| Evaluation Criterion | Candidate A: Inria Graphdeco (`diff-gaussian-rasterization`) | Candidate B: `gsplat` (Nerfstudio / UC Berkeley) |
| :--- | :--- | :--- |
| **Official Repository** | `github.com/graphdeco-inria/diff-gaussian-rasterization` | `github.com/nerfstudio-project/gsplat` |
| **Architecture** | Monolithic C++/CUDA extension; single entry point calls fused rasterizer. | Decoupled, modular PyTorch/CUDA library with distinct functional stages. |
| **Prebuilt Windows Wheels** | **None.** Must compile locally from source. | **Available.** Official prebuilt wheels hosted at `docs.gsplat.studio/whl`. |
| **Host Toolchain Requirements** | Requires full CUDA Toolkit (`nvcc`), MSVC C++ compiler, CMake, and `glm` submodule. | Only requires PyTorch with bundled CUDA runtime DLLs when using prebuilt wheels. |
| **Windows Friction** | **Extreme.** MSVC `14.51` on host has known compatibility conflicts with CUDA 12.x `host_config.h`. Missing local `nvcc`. | **Minimal.** Prebuilt wheels bypass local MSVC/NVCC compilation entirely. |
| **Tile Invalidation Feasibility** | **Poor.** Tile binning, radix sorting, and pixel loops are hardcoded inside CudaRasterizer without Python-accessible tile offsets. | **Ideal.** Clean entry points for tile overlap (`isect_tiles`), tile offsets (`isect_offset_encode`), and rasterization (`rasterize_to_pixels`). |
| **Maintenance & Ecosystem** | Infrequent updates; tightly coupled to original research code. | Actively maintained, optimized for PyTorch 2.x, widely adopted across academic research. |

### Technical Recommendation: `gsplat`
We strongly recommend **`gsplat`** with PyTorch 2.4.1 (CUDA 12.1 runtime).  
`gsplat` provides pre-compiled `win_amd64.whl` packages matching PyTorch 2.4 + CUDA 12.1, eliminating the requirement to install the multi-gigabyte CUDA Toolkit or debug MSVC compiler incompatibilities on Windows.

---

## 4. Proposed Packages, Resource Estimates & Safeguards

### Required Packages
1. **PyTorch with CUDA 12.1:**
   - Package: `torch==2.4.1`, `torchvision==0.19.1`
   - Source: Official PyTorch Index (`https://download.pytorch.org/whl/cu121`)
   - Bundled Components: Includes CUDA runtime DLLs (`cudart64_12.dll`, `cublas64_12.dll`, `cusparse64_12.dll`, `nvrtc64_120_0.dll`).
2. **Helper Dependencies:**
   - Packages: `jaxtyping`, `rich`, `ninja` (lightweight metadata and formatting utilities required by `gsplat`).
3. **Renderer:**
   - Package: `gsplat==1.4.0` (or compatible prebuilt build for PyTorch 2.4 / CUDA 12.1).
   - Source: `https://docs.gsplat.studio/whl/pt24cu121`

### Resource & Storage Estimates
- **Wheel Downloads:** ~2.5 GB total.
- **Extracted / Installed Size:** ~4.7 GB total.
- **Disk Allocation:** Configured strictly to `D:\pip_cache` and `D:\miniconda3\envs\3dgs_research`.
- **Impact on `C:\`:** Zero bytes added to `C:\`.
- **Impact on `D:\`:** Consumes ~5.0 GB out of 232.38 GB free (< 2.2% of available space).

### Windows Compatibility Risks & Mitigations
1. **Pip Cache Spill to `C:\`:**
   - *Risk:* Default pip caching writes multi-gigabyte wheels to `%LOCALAPPDATA%\pip\cache` on drive `C:`.
   - *Mitigation:* Explicitly enforce `--cache-dir D:\pip_cache` on every `pip install` invocation.
2. **NumPy 2.x Compatibility:**
   - *Status:* Environment currently has NumPy 2.4.6. PyTorch 2.4+ natively supports NumPy 2.x. If `gsplat` prebuilt wheel triggers a C-API version mismatch with NumPy 2.x, we will pin `numpy<2.0` as a safe fallback.
3. **Visual Studio C++ Runtime:**
   - *Status:* MSVC runtime libraries are already present on Windows 11. Prebuilt wheels link against standard CRT DLLs.

---

## 5. Minimal Reproducible Rendering Plan

To achieve the smallest reliable path to rendering a real Gaussian Splatting scene without downloading multi-gigabyte datasets (such as Mip-NeRF 360), we will execute the following controlled protocol:

### Step 1: Environment & GPU Verification (Zero Downloads)
- Install `torch==2.4.1` with CUDA 12.1 into `D:\miniconda3\envs\3dgs_research`.
- Execute a GPU tensor kernel (`randn` matrix multiplication) on `cuda:0`.
- Verify allocated VRAM and RTX 4060 detection.

### Step 2: Install Prebuilt `gsplat`
- Install `gsplat` via `--index-url https://docs.gsplat.studio/whl/pt24cu121`.
- Verify `import gsplat` and verify CUDA extension binding (`gsplat.csrc` availability).

### Step 3: Minimal Scene Construction (Self-Contained Synthetic 3DGS Scene)
Rather than downloading a large remote dataset, we will construct a small, mathematically complete 3D Gaussian Splatting scene directly in Python (e.g., 5,000–10,000 Gaussians representing two concentric colored geometric shapes):
- **Parameters Generated:**
  - 3D Means $\mu \in \mathbb{R}^{N \times 3}$
  - 3D Scales $s \in \mathbb{R}^{N \times 3}$ (log-scale)
  - Rotations $q \in \mathbb{R}^{N \times 4}$ (unit quaternions)
  - Opacities $\alpha \in \mathbb{R}^{N \times 1}$ (logit scale)
  - Colors/Spherical Harmonics $c \in \mathbb{R}^{N \times 1 \times 3}$ (RGB / SH degree 0)
- **Virtual Camera Setup:**
  - Resolution: $512 \times 512$
  - Focal length: $f_x = f_y = 512$
  - Principal point: $c_x = c_y = 256$
  - View matrix: Fixed extrinsic matrix pointing at origin.

### Step 4: Baseline Forward Render & Validation
- Execute `gsplat.rasterization` to project and render the scene.
- Save the output to `results/baseline_3dgs_render.png`.
- Measure baseline frame latency using `torch.cuda.Event`.
- Confirm visual correctness: Verify that overlapping Gaussians demonstrate proper depth ordering and continuous alpha compositing.

---

## 6. Tile-Level Incremental Rasterization Roadmap

Once the baseline renderer is operational, the research prototype will investigate partial screen-space updates. Below is the technical breakdown of the `gsplat` pipeline components and how they relate to incremental updates:

```
[3D Gaussian Parameters] (μ, s, q, α, c)
             │
             ▼
1. gsplat.project_gaussians / gsplat.rendering.rasterization
   ├── Transforms 3D covariance to 2D screen covariance Σ'
   ├── Calculates 3σ screen-space bounding radii (r)
   └── Outputs: means2d, conics, radii, depths
             │
             ▼
2. gsplat.isect_tiles
   ├── Maps each Gaussian's 2D bounding box to 16x16 screen tiles
   └── Generates 64-bit intersection keys: (tile_id << 32) | depth_key
             │
             ▼
3. CUB DeviceRadixSort
   └── Sorts Gaussian-tile pairs front-to-back per tile
             │
             ▼
4. gsplat.isect_offset_encode
   └── Produces `tile_offsets` of shape (num_tiles_y, num_tiles_x, 2)
       Each entry specifies [start_index, end_index] in sorted Gaussian array
             │
             ▼
5. gsplat.rasterize_to_pixels
   ├── Launches 1 CUDA thread block (16x16 threads) per tile
   └── Accumulates color front-to-back: C = Σ (c_i * α_i * T_i)
```

### Planned Modification Points for Incremental Rasterization:
1. **Upstream Invalidation (`src/dynamic/change_detector.py`):**
   - For any modified/moving Gaussian, compute its screen-space footprint at frame $t-1$ and frame $t$.
   - Generate a binary dirty tile mask $M_{\text{dirty}} \in \{0, 1\}^{(\lceil H/16 \rceil, \lceil W/16 \rceil)}$.
2. **Tile Offset Masking (`src/incremental/tile_manager.py`):**
   - In `tile_offsets`, zero out the ranges for clean tiles:  
     $$\text{tile\_offsets}[y, x] = [0, 0] \quad \forall (y, x) \notin M_{\text{dirty}}$$
   - This bypasses Gaussian loading and blending for clean tiles inside the rasterization kernel.
3. **Framebuffer Cache Blit (`src/incremental/cache_buffer.py`):**
   - Retain the previous frame's color tensor $I_{t-1}$.
   - Overwrite only the dirty $16 \times 16$ tile patches with newly rasterized tile data.
   - Preserves mathematical exactness because affected tiles re-evaluate the full depth-sorted stack of intersecting Gaussians.

---

## 7. Protection of Existing Prototypes

- [`src/change_detection.py`](file:///D:/3dgs-incremental-research/src/change_detection.py) and [`ui/app.py`](file:///D:/3dgs-incremental-research/ui/app.py) will remain untouched during this phase.
- Once the real 3DGS renderer baseline is verified, a new module [`src/baseline_3dgs.py`](file:///D:/3dgs-incremental-research/src/baseline_3dgs.py) will be added to the codebase.
- The Streamlit dashboard can subsequently be upgraded to display real 3DGS renders and dirty tile overlays side by side.

---

## 8. Immediate Action Items

1. **Stage 1A:** Install PyTorch 2.4.1 (CUDA 12.1) using the dedicated D: pip cache.
2. **Stage 1B:** Verify PyTorch GPU tensor execution on the RTX 4060.
3. **Stage 1C:** Install `gsplat` prebuilt wheel from `https://docs.gsplat.studio/whl/pt24cu121`.
4. **Stage 1D:** Render the minimal procedural 3DGS test scene to `results/baseline_3dgs_render.png`.
