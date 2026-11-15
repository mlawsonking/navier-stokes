import torch, math, time
from gpu_solver import ccf_sing as S, ccf_fast as F
DEV = "cuda"; sols = torch.load("runs/sing_track2_v8.pt"); x = sols[-1]; p = x["p"].to(DEV); mp = tuple(x["map"]); shape = (p.numel() - 34, 8, 4); n0 = shape[0]
print("state lam", p[0].item(), "mu", p[1].item(), "map", mp, "ypk", x["ypk"], "n0", n0, [ (s_["lam"], s_["p"][1].item()) for s_ in sols[-3:]])
F.MAP = mp; G = S.peak_grid(p[0].item(), x["ypk"], mp, n0, h=0.0125, R=6, device=DEV)
r = S.residual(p, G, shape); print("residual at stored state", r.abs().max().item(), "rows", G.y.numel())
for dm in (5e-5, 5e-4, 2e-3):
    g = p.clone(); g[1] = p[1] - dm; t = time.time()
    q, h = S.gn(g, G, shape, free=(True, False), max_iter=20); print("mu -", dm, "lam free ->", q[0].item(), [f"{v[1]:.1e}/{v[3]:.2f}" for v in h], f"{time.time()-t:.0f}s", flush=True)
