"""Which singular modes carry the floor residual, and is the linear model valid along them?  floor_diag6.py file.pt idx"""
import sys, math, torch, time
from gpu_solver import ccf_sing as S, ccf_fast as F
DEV = "cuda"; x = torch.load(sys.argv[1])[int(sys.argv[2])]
p = x["p"].to(DEV); mp = tuple(x["map"]); nj, J = x["njJ"]; n0 = p.numel() - 2 - nj * J; shape = (n0, nj, J); F.MAP = mp
G = S.peak_grid(p[0].item(), x["ypk"], mp, n0, h=0.0125, R=6, device=DEV)
r, Jm = S.res_jac(p, G, shape, (True, True)); d = Jm.norm(dim=0); U, sv, Vh = torch.linalg.svd((Jm / d).cpu(), full_matrices=False); rc = r.cpu(); co = U.T @ rc
print("col norms lam, mu:", d[0].item(), d[1].item())
order = torch.argsort(-(co.abs())); print("modes carrying the residual (index, sv/sv0, co, co/sv, v_lam/d, v_mu/d):")
for k in order[:10].tolist(): print(f"  {k:5d} {sv[k].item()/sv[0].item():.2e} {co[k].item():+.2e} {co[k].item()/sv[k].item():+.2e} {Vh[k,0].item()/d[0].item():+.2e} {Vh[k,1].item()/d[1].item():+.2e}")
BOm = S.operators(p[0], p[1], G, shape)[1]
for cut in (1e-3, 1e-5, 1e-7, 1e-9):
    k = sv > cut * sv[0]; st = (Vh[k].T @ (-co[k] / sv[k])).to(DEV) / d
    lin = (rc + (U[:, k] * sv[k]) @ (Vh[k] @ (st.cpu() * d.cpu()))).norm().item()
    out = []
    for t in (1.0, 0.1, 0.01):
        pn = p + t * st; out.append(f"t={t}: {S.residual(pn, G, shape).norm().item():.2e}")
    print(f"cut {cut:.0e} rank {int(k.sum())} dlam {st[0].item():+.2e} dmu {st[1].item():+.2e} dOm sup {(BOm @ st[2:]).abs().max().item():.2e} linear |r| {lin:.2e} actual " + " ".join(out))
# finite-difference check of the lam column against a much smaller step
for e in (1e-6, 1e-8):
    ee = torch.zeros_like(p); ee[0] = e; c1 = (S.residual(p + ee, G, shape) - S.residual(p - ee, G, shape)) / (2 * e)
    print(f"lam column FD e={e}: norm {c1.norm().item():.4e} diff from 1e-6 col {(c1 - Jm[:, 0]).norm().item():.2e}")
