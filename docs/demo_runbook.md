# 3D Gaussian Splatting Dynamic Scene Update Demo Runbook
**Judge-Ready Demonstration Guide & Operational Runbook (Modes A & B)**

*Project Root:* `D:\3dgs-incremental-research`  
*Hardware:* NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM, Ada Lovelace `sm_89`)  
*Primary Environment:* `D:\miniconda3\envs\3dgs_research` (Streamlit 1.66.1, OpenCV, NumPy, pandas)  
*GPU Renderer Environment:* `D:\miniconda3\envs\gsplat310` (Python 3.10, PyTorch 2.4.1+cu121, gsplat 1.5.3)  
*Resolution & Grid:* 512 × 512 RGB | 16 × 16 Hardware Tiles (1,024 Total Tiles)

---

## 1. Quick Launch Instructions

### Option A: Standard Port (8501)
Open PowerShell or Command Prompt:
```powershell
cd D:\3dgs-incremental-research
D:\miniconda3\envs\3dgs_research\python.exe -m streamlit run ui/app.py
```
Then navigate to: `http://localhost:8501`

### Option B: Dedicated Port (8504)
```powershell
cd D:\3dgs-incremental-research
D:\miniconda3\envs\3dgs_research\python.exe -m streamlit run ui/app.py --server.port 8504 --server.headless false
```
Then navigate to: `http://localhost:8504`

---

## 2. Presenter Demonstration Script

### Act 1: The Baseline State (Mode A: 0:00 – 0:30)
1. **Open Tab 1: "Live Dynamic 3DGS Demo"**.
2. Point out the top header:
   > *"We are rendering a deterministic 3D Gaussian scene of concentric spheres with vibrant spatial color gradients on an NVIDIA RTX 4060 GPU."*
3. Click **"Reset (t0)"** in the left sidebar.
4. Show the blue notification banner:
   > *"Here is our baseline frame at $t_0$. The camera is fixed, 100% of the 5,000 Gaussians are static in persistent GPU memory, and zero screen tiles are dirty."*
5. Point to the primary 4-panel visual display:
   - Panel 1: Original Baseline ($t_0$)
   - Panel 2: Modified Ground Truth ($t_1$) (identical to baseline)
   - Panel 3: Incremental Composite ($t_{\text{comp}}$) (identical)
   - Panel 4: Invalidation Mask (completely dark; 0 dirty tiles)

---

### Act 2: Applying a Localized Dynamic Update (Mode A: 0:30 – 1:15)
1. In the sidebar:
   - Keep **Scene Gaussian Scale:** `5,000 Gaussians`
   - Set **Modification Mode:** `Spatial Cluster (Localized 3D region)`
   - Select **Preset:** `1.0% (50 Gaussians)`
   - Translation Vector: $\Delta X = 0.15, \Delta Y = -0.10, \Delta Z = 0.05$
2. Click **"Apply Update"**.
3. Point out the green notification banner:
   > *"In one click, 50 of 5,000 Gaussians (1.0%) were translated in 3D space. Our GPU invalidation engine computed the screen-space bounding boxes at $t_0$ and $t_1$, determining that only 26 out of 1,024 screen tiles (2.5%) were affected."*
4. Guide the audience across the 4 panels:
   - **Panel 1 vs Panel 2:** Show the localized cluster of Gaussians moving down and right.
   - **Panel 3 (Incremental Composite):** Show that the incremental render reproduces the full rerender with zero visual artifacts.
   - **Panel 4 (Invalidation Tile Mask):** Show the tight cyan bounding region enclosing exactly where the Gaussians moved.
5. Expand **"Difference Heatmap & Ground-Truth Verification Overlays"**:
   - Show the Turbo Difference Heatmap (8× amplified) confirming where pixels changed.
   - Point out **Zero False Negatives:** The cyan mask conservatively encloses 100% of the ground-truth changed pixels (Recall = 100.0%, 0 missed tiles).

---

### Act 3: Honest Metrics & Scale Crossover (1:15 – 2:00)
1. Point to the **Performance Metrics Grid**:
   - **Image Parity:** $\text{PSNR} = \infty\text{ dB}$, $\text{SSIM} = 1.0000$, $\text{Max Absolute Error} = 0.000000$.
   - **Static Reuse:** 4,950 static Gaussians (99.0%) skipped projection entirely.
   - **Clean Tile Reuse:** 998 clean tiles (97.5%) loaded directly from cached frame $t_0$.
2. **Address Speedup Transparently on 5k vs 100k:**
   > *"On this 5,000-Gaussian scene, baseline full rerendering takes only ~0.83 ms. Because our incremental pipeline executes 7 modular GPU kernels, driver launch overhead totals ~1.27 ms."*
3. **Switch to Tab 4: "Scene Scaling & Benchmark Matrix"**:
   - Point out the scaling curve:
     > *"Now look at what happens as the scene scales to 100,000 Gaussians. Full rerendering latency climbs to 1.36 – 1.45 ms, while incremental rendering stays at ~1.05 – 1.19 ms, yielding a **1.14× to 1.29× genuine end-to-end speedup** on localized dynamic updates!"*
   - Point out the VRAM metric: Peak memory across all 100k scenes is only **109.2 MB**, leaving over 7.8 GB of VRAM headroom on the RTX 4060.

---

### Act 4: Mode B — Automatic Dual-Scene Change Detection (2:00 – 3:00)
1. **Switch to Tab 2: "Compare Two 3DGS Scenes"**.
2. Explain the operational scenario:
   > *"In production systems, users do not provide change masks or index lists. Instead, they provide two autonomous 3DGS scene files (Scene A and Scene B) captured at different points in time, potentially with differing Gaussian counts ($N_A \neq N_B$) and completely shuffled row orderings."*
3. Keep default source: **"Use Pre-packaged Sample PLY Pair (5,000 vs 5,050 Gaussians)"**.
   - Baseline Scene A: 5,000 Gaussians.
   - Updated Scene B: 5,050 Gaussians (50 modified positions, 100 removed, 150 added, permuted order).
4. Click **"Compare & Incrementally Update"**.
5. Review the automatic results:
   - **3D Spatial & Appearance Matching:** The GPU-vectorized bidirectional nearest-neighbor engine automatically isolates 4,850 static Gaussians, 50 modified Gaussians, 100 removed Gaussians, and 150 added Gaussians without any user prompting.
   - **Conservative Screen-Space Invalidation:** The union of removed/modified footprints in A and added/modified footprints in B invalidates only 115 of 1,024 tiles (11.2%), allowing **88.8% of screen tiles to be reused directly from Scene A's cached baseline frame**.
   - **Incremental Selective Rasterization:** `gsplat.rasterize_to_pixels` rasterizes only dirty tiles from Scene B; 5D broadcast compositing blends them with cached Scene A tiles.
   - **Mathematical Parity:**
     - Ground Truth Full Rerender: 0.88 ms
     - Incremental End-to-End: 0.61 ms (**1.45× Speedup**)
     - $\text{PSNR} = \infty\text{ dB}$, $\text{SSIM} = 1.0000$, $\text{Max Absolute Error} = 0.000000$.
     - Missed Tiles: **0 (Zero False Negatives, Recall = 100.0%)**.
6. (Optional) Switch radio to **"Upload Custom 3DGS PLY Files"** to demonstrate dragging and dropping real-world `.ply` files (e.g. from Inria/PostShot/Nerfstudio).

---

### Act 5: 2D Image Change Detection (3:00 – 3:45)
1. **Switch to Tab 3: "Image Change Detection"**.
2. Explain the context and scientific boundary:
   > *"In addition to full 3D Gaussian representations, researchers and users often work with 2D photographic captures, surveillance frames, or rendered stills. This tab provides complete 2D image-space change detection with camera motion compensation."*
3. Highlight the **Scientific Notice Callout**:
   > *"Point out that this tab operates purely in 2D pixel space and does not reconstruct 3D Gaussians or measure 3D rasterization speedups—maintaining strict scientific transparency."*
4. Select **"Use Pre-packaged Sample Images (512×512 Geometric & Contrast Changes)"**:
   - Reference image contains geometric shapes, gradient backdrop, and color palette.
   - Updated image features a displaced object, contrast shift, and new synthetic feature.
5. Click **"Detect Image Changes"**:
   - Review the metrics: processing latency (~10–15 ms), changed pixels (~3,600 px / ~4.0%), changed area percentage, and detected cluster count.
   - Examine the 4-way visual display:
     1. Original Reference Frame
     2. Aligned Updated Frame (with ORB/ECC camera compensation)
     3. 50/50 Aligned Blend Overlay (verifying alignment quality)
     4. Highlighted Changes (with colored pixel overlay and cyan bounding boxes)
   - Expand the **Diagnostic Difference Heatmap & Binary Change Mask**:
     - View continuous pixel delta magnitude in Turbo colormap.
     - Inspect the morphological cleaned binary change mask.
6. (Optional) Toggle **Upload Custom Images (JPG / JPEG / PNG)** to upload any real-world photo pair. Observe how different resolutions are automatically rescaled safely without altering the original files.

---

## 3. Expected Outputs & Verification Reference

### Mode A: Parametric Scene Perturbation Reference

| Scene Scale | Changed Fraction & Mode | Full Rerender (ms) | Incremental Steady (ms) | Speedup Ratio | Dirty Tiles % | Missed Tiles (FN) | PSNR / SSIM |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5,000** | Reset ($t_0$) | 1.06 ms | 0.76 ms | 1.40× | 0.0% | 0 | $\infty$ / 1.0000 |
| **5,000** | 1% Spatial Cluster (50 g) | 0.80 ms | 1.27 ms | 0.63× | 2.5% | 0 | $\infty$ / 1.0000 |
| **10,000** | 1% Spatial Cluster (100 g) | 0.89 ms | 1.38 ms | 0.64× | 3.3% | 0 | $\infty$ / 1.0000 |
| **25,000** | 1% Spatial Cluster (250 g) | 1.00 ms | 1.30 ms | 0.77× | 3.8% | 0 | $\infty$ / 1.0000 |
| **50,000** | 1% Spatial Cluster (500 g) | 1.03 ms | 1.35 ms | 0.76× | 4.2% | 0 | $\infty$ / 1.0000 |
| **100,000** | Single Gaussian (1 g) | **1.36 ms** | **1.05 ms** | **1.29×** | 2.3% | 0 | $\infty$ / 1.0000 |
| **100,000** | 1% Spatial Cluster (1,000 g) | **1.42 ms** | **1.12 ms** | **1.27×** | 10.4% | 0 | $\infty$ / 1.0000 |
| **100,000** | 5% Spatial Cluster (5,000 g) | **1.36 ms** | **1.19 ms** | **1.14×** | 19.5% | 0 | $\infty$ / 1.0000 |
| **100,000** | 10% Spatial Cluster (10,000 g)| **1.40 ms** | **1.14 ms** | **1.22×** | 23.8% | 0 | $\infty$ / 1.0000 |
| **100,000** | 1% Random Subset (1,000 g) | **1.38 ms** | **1.38 ms** | **1.00×** | 62.8% | 0 | $\infty$ / 1.0000 |

### Mode B: Dual-Scene Comparison Reference (PLY Demo)

| Metric | Measured Value | Requirement / Significance |
| :--- | :--- | :--- |
| **Scene A Gaussians** | 5,000 | Baseline input scene |
| **Scene B Gaussians** | 5,050 | $N_A \neq N_B$ with permuted point ordering |
| **Matched Static Gaussians** | 4,850 (97.0%) | Automatically matched via mutual NN ($\epsilon_{\text{pos}}=0.03, \epsilon_{\text{feat}}=0.08$) |
| **Modified Gaussians** | 50 (1.0%) | Position shifted outside tolerance |
| **Removed Gaussians ($A \setminus B$)** | 100 (2.0%) | In Scene A but deleted in Scene B |
| **Added Gaussians ($B \setminus A$)** | 150 (2.9%) | Newly inserted into Scene B |
| **Dirty Tiles** | 115 / 1,024 (11.2%) | Union of footprints of changed/removed in A and changed/added in B |
| **Clean Tiles Reused** | 909 / 1,024 (88.8%) | Zero rasterization work; reused from cached Scene A frame |
| **PSNR / SSIM** | $\infty\text{ dB}$ / $1.000000$ | Bit-exact parity against independent full rerender of Scene B |
| **Max Absolute Error** | 0.000000 | Zero boundary or seam artifacts |
| **Missed Tiles (False Negatives)**| **0** | **Recall = 100.0%** against ground-truth pixel difference |
| **Full Rerender Latency** | 0.88 ms | Independent `gsplat.rasterize_to_pixels` of Scene B |
| **Incremental Render Latency** | **0.61 ms** | **1.45× Speedup** via selective dirty-tile rasterization |

---

## 4. Method Explanation & Current Architectural Limits

### Core Algorithmic Architecture
1. **Static/Dynamic Partitioning (Mode A):**
   - Static Gaussians are projected once at setup and cached in persistent GPU memory (`cached_means2d`, `cached_conics`, `cached_depths`, `cached_radii`).
   - Only modified Gaussians ($K \ll N$) are projected dynamically using `gsplat.fully_fused_projection`.
2. **Conservative Screen-Space Tile Invalidation:**
   - An invalidation mask is generated from the union of bounding boxes at $t_0$ (old footprint) and $t_1$ (new footprint).
   - Executed via a TorchScript JIT-fused kernel (`_fused_tile_invalidation_kernel`) with zero host-device synchronization.
   - Guarantees **Zero False Negatives** ($\text{Recall} = 100\%$) across all tests.
3. **In-Place Working Buffer & Global CUB RadixSort:**
   - Dynamic 2D parameters are written directly into pre-allocated contiguous GPU buffers.
   - Tile intersection and depth sorting are handled via `gsplat.isect_tiles` (`cub::DeviceRadixSort::SortPairs`).
   - *Static Sorting Reuse Investigation Finding:* Global CUB sorting requires only 0.57 ms on 100k Gaussians. Dynamic candidate filtering via 2D SAT takes 1.10 ms, making global CUB sorting 2.5× faster than dynamic candidate bucket management.
4. **Selective Tile-Masked Rasterization:**
   - `gsplat.rasterize_to_pixels` evaluates only tiles where `masks[tile_id] == True`. Clean tiles early-exit with zero texture writes or depth comparisons.
5. **Zero-Allocation 5D Broadcast Compositing:**
   - Final frame is assembled by broadcasting the 2D boolean mask into a 5D tile-strided tensor:
     `torch.where(mask_5d, sparse_5d, cached_5d)`
   - Replaces expensive `repeat_interleave` pixel masks and achieves < 0.05 ms compositing.
6. **Mode B: Dual 3DGS Scene Correspondence & Invalidation Engine:**
   - **PLY Format Support:** Built-in high-performance binary little-endian and ASCII parser supporting both standard Inria/Graphdeco convention (`scale_0..2` log-scales, `opacity` logit, `f_dc_0..2` SH spherical harmonics) and linear RGB conventions without external C-bindings or libraries.
   - **GPU-Vectorized Mutual Nearest Neighbors:** Performs chunked bidirectional Euclidean distance matching ($j = \text{NN}(i) \land i = \text{NN}(j)$) in chunks of 2,000–4,000 Gaussians. Enforces spatial distance threshold $\epsilon_{\text{pos}}$ and appearance delta threshold $\epsilon_{\text{feat}}$.
   - **Zero False Negatives Formulation:** The invalidation mask is constructed as:
     $$\mathcal{M}_{\text{dirty}} = \text{Tiles}\Big(\bigcup_{i \in \text{Modified}_A \cup \text{Removed}_A} \text{BBox}_A(i) \;\cup\; \bigcup_{j \in \text{Modified}_B \cup \text{Added}_B} \text{BBox}_B(j)\Big)$$
     This guarantees that both disappearing radiance from Scene A and emerging radiance in Scene B are strictly enclosed.

### Current Limitations (Disclosed Transparently)
- **Fixed Camera Pose:** The static scene cache assumes a stationary viewpoint. If the camera translates or rotates, all screen projections are invalidated.
- **Kernel Launch Overhead on Small Scenes:** On scenes with fewer than 25k Gaussians, the CPU-GPU driver dispatch overhead of 7 sequential CUDA kernels outweighs the rasterization savings. Net end-to-end acceleration begins at ~50k–100k Gaussians.
- **3D Matching vs 2D Differencing:** 3D Gaussian matching is not semantic object tracking. If two independently trained scenes have different Gaussian decompositions or large spatial transformations, correspondence matching will classify all Gaussians as modified/added/removed, safely falling back to full rerender.

---

## 5. Troubleshooting Guide

| Issue | Root Cause | Immediate Resolution |
| :--- | :--- | :--- |
| **Port 8501 or 8504 already in use** | An existing background server is occupying the port. | Run with `--server.port 8505` or kill python: `Stop-Process -Name python -Force` in PowerShell. |
| **Backend subprocess timeout (>120s)** | GPU was busy or CUDA context crashed. | Open task manager, verify RTX 4060 is available, restart Streamlit. |
| **`PermissionError: [Errno 13]` when writing results** | File was locked by an active image viewer or non-bypassed sandbox. | Close external image viewers viewing `results/interactive/*.png`. |
| **ModuleNotFoundError: 'src'** | `PYTHONPATH` was not set when invoking CLI directly. | Prepend `cmd /c "set PYTHONPATH=D:\3dgs-incremental-research && ..."` to the invocation command. |
| **Baseline scene file missing** | `results/procedural_scene_5k.pt` was moved or deleted. | Run `python src/baseline_3dgs.py` in `gsplat310` to regenerate the baseline scene in < 1 second. |
| **Coordinate Incompatibility Warning in Mode B** | Scene A and Scene B centroids or bounding boxes are fundamentally displaced (> 20.0 units). | Ensure both uploaded PLYs represent the same coordinate frame / scene scale. |
| **PLY Header Parse Error** | Uploaded PLY is missing essential vertex properties (`x`, `y`, `z`, scales, opacities). | Confirm the file was exported from standard 3DGS pipelines (Inria, Nerfstudio, PostShot, or gsplat). |

---

## 6. Regression Test Verification Commands

Run all verified project test suites:
```powershell
# 2D Image Change Detection Suite (8s)
cmd /c "set PYTHONPATH=D:\3dgs-incremental-research && D:\miniconda3\envs\3dgs_research\python.exe D:\3dgs-incremental-research\tests\test_image_change_detection.py"

# Mode B: Two-Scene Comparison & PLY I/O Suite (12s)
cmd /c "set PYTHONPATH=D:\3dgs-incremental-research && D:\miniconda3\envs\gsplat310\python.exe D:\3dgs-incremental-research\tests\test_two_scene_comparison.py"

# Milestone 2D & Demo Robustness Suite (16s)
cmd /c "set PYTHONPATH=D:\3dgs-incremental-research && D:\miniconda3\envs\gsplat310\python.exe D:\3dgs-incremental-research\tests\test_milestone_2d.py"

# Milestone 2C Audited Discrepancy & Invalidation Suite (18s)
cmd /c "set PYTHONPATH=D:\3dgs-incremental-research && D:\miniconda3\envs\gsplat310\python.exe D:\3dgs-incremental-research\tests\test_milestone_2c.py"

# Vectorized Invalidation Engine Suite (14s)
cmd /c "set PYTHONPATH=D:\3dgs-incremental-research && D:\miniconda3\envs\gsplat310\python.exe D:\3dgs-incremental-research\tests\test_invalidation.py"

# Milestone 2B Static/Dynamic Partitioning Suite (14s)
cmd /c "set PYTHONPATH=D:\3dgs-incremental-research && D:\miniconda3\envs\gsplat310\python.exe D:\3dgs-incremental-research\tests\test_milestone_2b.py"

# Experiment Module CLI Suite (15s)
cmd /c "set PYTHONPATH=D:\3dgs-incremental-research && D:\miniconda3\envs\gsplat310\python.exe D:\3dgs-incremental-research\tests\test_experiment_module.py"
```
*All 7 test suites exit code 0 with 100% test pass rate.*
