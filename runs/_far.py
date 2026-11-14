import torch, math, time, sys
from gpu_solver import ccf as C, ccf_fast as F
torch.set_num_threads(7)
F.MODE.update(norm="origin", origin_row=True, d=0.1, ymin=0.0)
sols = torch.load("runs/gclm_u1_branch_upc288.pt")
for at in (-0.52, -0.60):
    x = [t for t in sols if abs(t['a'] - at) < 1e-9][0]; p = x['p']; F.MAP = tuple(x['map']); n = p.numel() - 1
    G = C.Grid(h=0.0125, L=40.0, Lc=22.0, device="cpu", n_gl=128)
    for far in (None, (6, 1), (6, 2), (10, 3)):
        F.FAR = far; g = p if far is None else torch.cat([p, torch.zeros(far[0] * far[1], dtype=C.DT)])
        t = time.time(); q, h = F.lm(g, G, a=at, lam_free=True, max_iter=80, mu0=1e-6, stall=5); c = q[1:n + 1].abs()
        print(f"a {at} FAR {far} lam {q[0].item():.12f} sup {h[-1][1]:.1e} its {len(h)} far {q[n+1:].abs().max().item() if far else 0:.1e} coef", " ".join(f"{c[k:k+36].max().item():.0e}" for k in range(0, n, 36)), f"{time.time()-t:.0f}s", flush=True)
