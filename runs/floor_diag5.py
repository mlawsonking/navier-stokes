"""Stationarity of the floor states and a both-free solve.  floor_diag5.py file.pt idx"""
import sys, math, torch, time
from gpu_solver import ccf_sing as S, ccf_fast as F
DEV = "cuda"; x = torch.load(sys.argv[1])[int(sys.argv[2])]
p = x["p"].to(DEV); mp = tuple(x["map"]); nj, J = x["njJ"]; n0 = p.numel() - 2 - nj * J; shape = (n0, nj, J); F.MAP = mp
G = S.peak_grid(p[0].item(), x["ypk"], mp, n0, h=0.0125, R=6, device=DEV)
r, Jm = S.res_jac(p, G, shape, (True, True)); d = Jm.norm(dim=0); U, sv, Vh = torch.linalg.svd((Jm / d).cpu(), full_matrices=False); rc = r.cpu()
co = U.T @ rc; print(f"|r| {rc.norm().item():.3e}  sv max {sv[0].item():.2e} min {sv[-1].item():.2e}")
for cut in (1e-6, 1e-9, 1e-12, 1e-15, 0):
    k = sv > cut * sv[0]; print(f"  cut {cut:.0e} rank {int(k.sum())} |P r|/|r| {co[k].norm().item()/rc.norm().item():.3e}  step sup {(Vh[k].T @ (co[k]/sv[k]) / d.cpu()).abs().max().item():.2e}")
t0 = time.time(); q, hist = S.gn(p, G, shape, free=(True, True), max_iter=10, rc=1e-9, rc0=1e-9, floor=1e-12, verbose=True)
print(f"both free: lam {p[0].item():.9f} -> {q[0].item():.9f} mu {p[1].item():.9f} -> {q[1].item():.9f} sup {S.residual(q, G, shape).abs().max().item():.2e} t={time.time()-t0:.0f}s")
