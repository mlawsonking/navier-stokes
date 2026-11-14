"""Exact check of the general-a solver. For a = 1/2 the profile equation has the exact solution
    lam = -2/3,  Omega = -(16/3) y/(1+y^2)^2,  U = (8/3) y/(1+y^2)      (verified by hand; G = -(8/3)/(y+i)^2)
and any rescaling Omega(c y). In the basis: alpha = 3, beta = -1, Theta = (8/3)/r up to a constant, i.e. M = const.
The solver is started away from it (lam0 = -0.5, perturbed coefficients) with lam unknown."""
import torch
from gpu_solver import ccf as C, ccf_fast as F
F.MAP = None; F.MODE.update(norm="origin", origin_row=True, d=16.0/3.0, ymin=0.0)
for n, h in ((12, 0.05), (24, 0.025)):
    G = C.Grid(h=h, L=40.0, Lc=14.0, device="cpu")
    torch.manual_seed(0)
    p = torch.cat([torch.tensor([-0.66], dtype=C.DT), torch.tensor([2.66], dtype=C.DT), 0.002 * torch.randn(n - 1, dtype=C.DT)])
    q, hist = F.lm(p, G, a=0.5, max_iter=200)
    r = F.residual(q[0], q[1:], G, 0.5)
    print(f"n={n} h={h}: lam {q[0].item():.15f} (exact -0.666666666666667) err {abs(q[0].item()+2/3):.1e} sup {r.abs().max().item():.1e} its {len(hist)} c0 {q[1].item():.12f} |c_rest| {q[2:].abs().max().item():.1e}")
