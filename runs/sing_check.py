"""ccf_sing sanity: the smooth first unstable profile must satisfy the new discretisation (mu = 2)."""
import torch, time
from gpu_solver import ccf_sing as S, ccf_fast as F
DEV = "cuda"
p_old = torch.load("runs/ccf_u1_v_160_0.0125_0.1.pt").reshape(-1)
old = (-0.14, 0.1)
for shape in [(160, 8, 0), (160, 8, 4)]:
    G = S.EGrid(h=0.0125, device=DEV)
    p, M1 = S.from_smooth(p_old, old, old, shape, DEV)
    r = S.residual(p, G, shape)
    print(shape, "M(1) of old rep", M1, "rows", r.numel(), "residual sup", r[:-2].abs().max().item(), "norm,origin rows", r[-2:].tolist())
    y = G.y; rr = r[:-2].abs()
    for lo, hi in [(0, 1e-4), (1e-4, 1e-2), (1e-2, 1), (1, 10), (10, 1e4), (1e4, 1e10)]:
        k = (y >= lo) & (y < hi); print(f"   y in [{lo},{hi}) max {rr[k].max().item():.1e}")
    t0 = time.time()
    q, hist = S.lm(p, G, shape, free=(True, False), max_iter=40)
    print("   smooth solve: lam", q[0].item(), "sup", hist[-1][1], "its", len(hist), f"{time.time()-t0:.0f}s")
