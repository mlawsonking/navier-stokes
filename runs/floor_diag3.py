import sys, math, torch
from gpu_solver import ccf_sing as S, ccf_fast as F
DEV = "cuda"; x = torch.load(sys.argv[1])[int(sys.argv[2])]
p = x["p"].to(DEV); mp = tuple(x["map"]); nj, J = x["njJ"]; n0 = p.numel() - 2 - nj * J; shape = (n0, nj, J); lam = p[0].item(); F.MAP = mp
for h, R, sh in [(0.0125, 6, 0.0), (0.01, 7.8, 0.37)]:
    G = S.peak_grid(lam, x["ypk"], mp, n0, h=h, R=R, device=DEV, shift=sh)
    BTh, BOm, AU, up0, wt, nvec = S.operators(p[0], p[1], G, shape); c = p[2:]
    Th, Om, U = BTh @ c, BOm @ c, AU @ c; V = (1 + lam) * G.y - U; Rr = (-lam * Th + V * Om) / wt
    i = Om.abs().argmax().item(); print(f"h {h} peak row {i} y {G.y[i].item():.6f} spacing {(G.y[i+1]-G.y[i]).item():.2e} V* {V[i].item():.6e} wt {wt[i].item():.3f}")
    for k in range(i - 60, i + 61, 6):
        print(f"   dy {(G.y[k]-G.y[i]).item():+.3e} Om {Om[k].item():9.3f} V {V[k].item():.6e} R {Rr[k].item():+.3e} R/Om^2 {Rr[k].item()/Om[k].item()**2:+.3e} dV=R*wt/Om {Rr[k].item()*wt[k].item()/Om[k].item():+.3e}")
