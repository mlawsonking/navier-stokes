import torch, math, time, sys
from gpu_solver import ccf_sing as S, ccf_fast as F
DEV = "cuda"; x = torch.load("runs/sing_track2_v2.pt")[-1]; p = x["p"].to(DEV); F.MAP = tuple(x["map"]); shape = (640, 8, 4)
G = S.EGrid(h=0.0125, device=DEV, cluster=("box", math.log(x["ypk"]), 0.15, 80.0))
g = p.clone(); g[0] = p[0] - 0.0003
for rc in (1e-9, 1e-8, 1e-10):
    t = time.time(); q, h = S.gn(g, G, shape, rc=rc); print("QR-Tikhonov rc", rc, "its", len(h), [f"{v[1]:.0e}/{v[3]:.2f}" for v in h], "mu", q[1].item(), f"{time.time()-t:.0f}s", flush=True)
