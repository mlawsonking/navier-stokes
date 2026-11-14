"""Representation test: how well does the basis hold the first unstable profile under the scaling
Omega(y) -> Omega(c y) (i.e. normalisation Omega'(0) = -d)? Least-squares fit error vs n and d."""
import torch
from gpu_solver import ccf as C, ccf_fast as F
p = torch.load("runs/ccf_u1_g.pt").cpu().reshape(-1)
lam = p[0]
G = C.Grid(h=0.0125, L=40.0, Lc=22.0, device="cpu")
al = (1 / (1 + lam)).item()
for d in (0.005, 0.02, 0.05, 0.1, 0.3, 1.0, 3.0):
    c = d / 0.1
    Og = C.fields_local(p, c * G.y)[1]
    wts = 1 / (G.y * (1 + G.y ** 2) ** (-(1 + al) / 2))
    row = f"d={d:<6}"
    for n in (64, 128, 192):
        B = F.operators(lam, G, n)[0]
        cf = torch.linalg.lstsq(B * wts[:, None], (Og * wts)[:, None]).solution[:, 0]
        row += f"  n={n}: {((B @ cf - Og) * wts).abs().max().item():.1e}"
    print(row, flush=True)
