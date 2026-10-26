import sys
from pathlib import Path
ROOT = Path(r"D:\3dgs-incremental-research")
sys.path.insert(0, str(ROOT))

import torch

device = torch.device("cuda:0")
na = 5000
nb = 5050
pos_a = torch.randn(na, 3, device=device)
pos_b = torch.randn(nb, 3, device=device)

# Method 1: torch.cdist
torch.cuda.synchronize()
ev_s = torch.cuda.Event(enable_timing=True)
ev_e = torch.cuda.Event(enable_timing=True)

# Warmup
for _ in range(5):
    _ = torch.cdist(pos_a, pos_b)
torch.cuda.synchronize()

reps = 30
ev_s.record()
for _ in range(reps):
    dists = torch.cdist(pos_a, pos_b)
    m_dist_b, m_idx_b = torch.min(dists, dim=-1)
    dists_rev = torch.cdist(pos_b, pos_a)
    m_dist_a, m_idx_a = torch.min(dists_rev, dim=-1)
ev_e.record()
torch.cuda.synchronize()
t_cdist = ev_s.elapsed_time(ev_e) / reps
print(f"torch.cdist (fwd + rev): {t_cdist:.4f} ms")

# Method 2: Tensor-Core torch.mm for squared Euclidean distance
# ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x @ y.T
ev_s.record()
for _ in range(reps):
    norm_a = (pos_a ** 2).sum(dim=-1, keepdim=True)  # [na, 1]
    norm_b = (pos_b ** 2).sum(dim=-1, keepdim=True)  # [nb, 1]
    # gram matrix
    gram = torch.mm(pos_a, pos_b.t())                # [na, nb] via Tensor Cores!
    # dist_sq = norm_a + norm_b.t() - 2 * gram
    dist_sq_ab = torch.addmm(norm_a + norm_b.t(), pos_a, pos_b.t(), alpha=-2.0, beta=1.0)
    m_dist_sq_b, m_idx_b = torch.min(dist_sq_ab, dim=-1)
    # Reverse is just transpose! ZERO EXTRA DISTANCE COMPUTATION!
    m_dist_sq_a, m_idx_a = torch.min(dist_sq_ab, dim=0)
ev_e.record()
torch.cuda.synchronize()
t_mm = ev_s.elapsed_time(ev_e) / reps
print(f"torch.addmm Tensor Core + Transpose reuse: {t_mm:.4f} ms")
print(f"Speedup: {t_cdist / t_mm:.2f}x faster!")
