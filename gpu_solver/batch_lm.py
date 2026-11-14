"""Batched Levenberg-Marquardt: B independent small least-squares problems solved
at once on the GPU (multi-start, parameter scans). Same step as lm.py (column-scaled
QR of the damped Jacobian), with per-problem damping and acceptance.
"""
import torch
from torch.func import jacfwd, jacrev, vmap


def batched_lm(resid, P0, aux=None, max_iter=300, tol=1e-14, mu0=1e-3, chunk=64, stall=10, fwd=False):
    """resid(p[, aux_i]) -> [m] for one problem. P0: [B, P]. aux: [B, ...] or None.
    Returns p [B,P], sup residual [B], iterations used [B]."""
    jac = (lambda f, **k: jacfwd(f, **{q: v for q, v in k.items() if q != "chunk_size"})) if fwd else jacrev
    if aux is None:
        rb, Jb = vmap(resid), vmap(jac(resid, chunk_size=chunk))
        R = lambda p: rb(p); Jf = lambda p: Jb(p)
    else:
        rb, Jb = vmap(resid), vmap(jac(resid, argnums=0, chunk_size=chunk))
        R = lambda p: rb(p, aux); Jf = lambda p: Jb(p, aux)
    p = P0.clone()
    B, P = p.shape
    r = R(p)
    cost = 0.5 * (r * r).sum(1)
    mu = torch.full((B,), mu0, dtype=p.dtype, device=p.device)
    active = torch.isfinite(cost)
    iters = torch.zeros(B, dtype=torch.long, device=p.device)
    eye = torch.eye(P, dtype=p.dtype, device=p.device).expand(B, P, P)
    zeros = torch.zeros(B, P, dtype=p.dtype, device=p.device)
    slow = torch.zeros(B, dtype=torch.long, device=p.device)
    for it in range(max_iter):
        cost_prev = cost
        J = torch.nan_to_num(Jf(p))
        d = J.norm(dim=1).clamp_min(1e-30)
        Js = J / d[:, None, :]
        rhs = torch.cat([-r, zeros], 1)[..., None]
        done = ~active
        tiny = torch.zeros(B, dtype=torch.bool, device=p.device)
        for _ in range(30):
            Q, Rm = torch.linalg.qr(torch.cat([Js, mu.sqrt()[:, None, None] * eye], 1))
            step = torch.linalg.solve_triangular(Rm, Q.transpose(1, 2) @ rhs, upper=True)[..., 0] / d
            pn = p + step
            rn = R(pn)
            cn = 0.5 * (rn * rn).sum(1)
            ok = torch.isfinite(cn) & (cn < cost) & ~done
            p = torch.where(ok[:, None], pn, p)
            r = torch.where(ok[:, None], rn, r)
            cost = torch.where(ok, cn, cost)
            tiny |= ok & (step.abs().amax(1) < 1e-15)
            mu = torch.where(ok, (mu / 3).clamp_min(1e-15), torch.where(done, mu, mu * 4))
            done = done | ok
            if done.all():
                break
        iters += active.long()
        # stalled: relative cost decrease below 1e-6 for `stall` iterations in a row
        slow = torch.where(cost > cost_prev * (1 - 1e-6), slow + 1, torch.zeros_like(slow))
        active = active & done & ~tiny & (r.abs().amax(1) >= tol) & (slow < stall)
        if not active.any():
            break
    return p, r.abs().amax(1), iters
