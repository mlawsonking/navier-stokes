"""Random multi-start at fixed lam (k=1 family). Usage: ccf_multistart.py n h lam nstarts device seed"""
import sys, time, torch
from gpu_solver import ccf as C, ccf_fast as F
n, h, lam, ns, dev, seed = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5], int(sys.argv[6])
import os
if os.environ.get('FREE'): F.MODE.update(norm='tail', origin_row=False)
G = C.Grid(h=h, L=40.0, Lc=22.0, device=dev)
torch.manual_seed(seed)
k = torch.arange(1, n + 1, dtype=C.DT, device=dev)
best = []
t0 = time.time()
for i in range(ns):
    amp = [0.3, 1.0, 3.0][i % 3]
    c = amp * torch.randn(n, dtype=C.DT, device=dev) / k ** 2
    c[0] += (-1.0 if os.environ.get('FREE') else -1 / ((1 - 1 / (1 + lam)) / 2))
    p, hist = F.lm(torch.cat([torch.tensor([lam], dtype=C.DT, device=dev), c]), G, lam_free=False, max_iter=150)
    r = F.residual(p[0], p[1:], G)
    best.append((r.norm().item(), i, p))
    BOm, BOmy, AU, AUp, up0, q1, wt = F.operators(p[0], G, n)
    print(f"   Om'(0)={(q1 @ p[1:]).item():+.4f} U'(0)={(up0 @ p[1:]).item():.5f} (mu=2 needs {1+lam/2:.5f}, mu=4 needs {1+0.75*lam:.5f})")
    print(f"start {i} amp {amp} l2 {r.norm().item():.3e} sup {r.abs().max().item():.2e} its {len(hist)} c0..3 {[round(x,3) for x in p[1:5].tolist()]}", flush=True)
best.sort(key=lambda t: t[0])
print("best:", [(f"{b[0]:.2e}", b[1]) for b in best[:8]], f"wall {time.time()-t0:.0f}s")
torch.save(torch.stack([b[2] for b in best[:8]]), f"runs/ccf_ms_lam{lam}_n{n}.pt")
