import sys
from pathlib import Path
ROOT = Path(r"D:\3dgs-incremental-research")
sys.path.insert(0, str(ROOT))

import torch
from src.scene_cache import _fused_tile_invalidation_kernel

device = torch.device("cuda:0")
K = 348
m2d = torch.randn(K, 2, device=device) * 200 + 256
rad = torch.abs(torch.randn(K, 2, device=device)) * 20 + 5
dep = torch.ones(K, device=device) * 2.0
y_grid = torch.arange(32, device=device).view(32, 1)
x_grid = torch.arange(32, device=device).view(1, 32)

# Warmup JIT
for _ in range(5):
    _ = _fused_tile_invalidation_kernel(m2d, rad, dep, y_grid, x_grid, 512, 512, 16, 32, 32)
torch.cuda.synchronize()

ev_s = torch.cuda.Event(enable_timing=True)
ev_e = torch.cuda.Event(enable_timing=True)
reps = 100
ev_s.record()
for _ in range(reps):
    _ = _fused_tile_invalidation_kernel(m2d, rad, dep, y_grid, x_grid, 512, 512, 16, 32, 32)
ev_e.record()
torch.cuda.synchronize()
print(f"JIT Kernel avg latency over {reps} reps: {ev_s.elapsed_time(ev_e) / reps:.4f} ms")
