"""Levenberg-Marquardt for small dense least-squares problems, float64 on CUDA.

Jacobians come from torch.func.jacrev, so the residual only has to be a pure
torch function of the parameter vector.
"""
import math
import torch
from torch.func import jacrev


def levenberg_marquardt(resid, p0, max_iter=200, tol=1e-14, mu0=1e-3, verbose=False):
    p = p0.clone()
    r = resid(p)
    cost = 0.5 * (r @ r)
    mu = mu0
    J_of = jacrev(resid)
    hist = []
    for it in range(max_iter):
        J = J_of(p)
        g = J.T @ r
        # column scaling + damped least squares solved by QR (never forms J^T J,
        # which would square the condition number and cap accuracy near 1e-8)
        d = J.norm(dim=0).clamp_min(1e-30)
        Js = J / d
        improved = False
        for _ in range(30):
            Q, Rm = torch.linalg.qr(torch.cat([Js, math.sqrt(mu) * torch.eye(Js.shape[1], dtype=J.dtype, device=J.device)]))
            rhs = torch.cat([-r, torch.zeros(Js.shape[1], dtype=r.dtype, device=r.device)])
            step = torch.linalg.solve_triangular(Rm, (Q.T @ rhs)[:, None], upper=True)[:, 0] / d
            pn = p + step
            rn = resid(pn)
            cn = 0.5 * (rn @ rn)
            if torch.isfinite(cn) and cn < cost:
                p, r, mu = pn, rn, max(mu / 3, 1e-15)
                improved = cost - cn
                cost = cn
                break
            mu *= 4
        sup = r.abs().max().item()
        hist.append((it, cost.item(), sup, mu))
        if verbose:
            print(f"it {it:3d} cost {cost.item():.3e} sup {sup:.3e} mu {mu:.1e}")
        if improved is False or sup < tol or step.abs().max().item() < 1e-15:
            break
    return p, hist
