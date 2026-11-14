"""CCF profiles with Omega ~ -|y|^nu sgn(y) at the origin (nu >= 1 real).

Local analysis at y = 0 (derived in this project; a = -1): if Omega ~ |y|^nu sgn(y) then
    nu = (U'(0) - 1) / (1 + lam - U'(0)),
the nonlocal term entering only at higher order. nu = 1 is the stable profile's family (then
lam = 2U'(0) - 2); a profile is smooth at the origin iff nu is an odd integer. This module solves for
(lam, c) at prescribed nu, so nu can be continued from 1 to 3, 5, ...

    alpha = 1/(1+lam), beta = (1-alpha)/2, r = 1+y^2, s = r^(-alpha/2), kappa = (nu+1)/2
    Theta = r^beta (1-s)^kappa N(s),  N = sum c_k T_k(2s-1)      (even; ~ y^(nu+1) at 0; tail on the s-lattice)
Rows: collocation, N(1) = -1 (scale), and the origin relation above. Quadrature as in ccf.py.
"""
import math
import torch
from .ccf import cheb3, Grid, DT


def _basis(lam, kappa, y, n, second=True):
    alpha = 1.0 / (1.0 + lam); beta = (1 - alpha) / 2
    r = 1 + y * y
    s = r ** (-alpha / 2)
    T, d1, d2 = cheb3(2 * s - 1, n)
    N1, N2 = 2 * d1, 4 * d2
    e = (1 - s).clamp_min(0.0)[:, None]
    pk1 = e ** (kappa - 1)
    M = pk1 * e * T
    M1 = pk1 * (-kappa * T + e * N1)
    sc = s[:, None]
    Q = 2 * beta * M - alpha * sc * M1
    BTh = (r ** beta)[:, None] * M
    BOm = (y * r ** (beta - 1))[:, None] * Q
    if not second:
        return BTh, BOm, None, Q, alpha, beta
    M2 = e ** (kappa - 2) * (kappa * (kappa - 1) * T - 2 * kappa * e * N1 + e * e * N2)
    Q1 = (2 * beta - alpha) * M1 - alpha * sc * M2
    BOmy = (r ** (beta - 1))[:, None] * Q + (y * y * r ** (beta - 2))[:, None] * (2 * (beta - 1) * Q - alpha * sc * Q1)
    return BTh, BOm, BOmy, Q, alpha, beta


def operators(lam, kappa, G, n):
    BTh_s, BOm_s, _, Q_s, alpha, beta = _basis(lam, kappa, G.ys, n, second=False)
    _, BOm, BOmy, _, _, _ = _basis(lam, kappa, G.y, n)
    sY = (1 + G.Y * G.Y) ** (-alpha / 2)
    sg = sY * G.g
    Tg, d1g, _ = cheb3(2 * sg - 1, n)
    eg = (1 - sg)[:, None]
    Mg = eg ** kappa * Tg
    M1g = eg ** (kappa - 1) * (-kappa * Tg + eg * 2 * d1g)
    tM = sY * (G.gw @ Mg) / alpha
    tQ = sY * (G.gw @ (2 * beta * Mg - alpha * sg[:, None] * M1g)) / alpha
    AU = (2 * G.y / math.pi)[:, None] * (G.K @ BTh_s - tM)
    AUp = (2 / math.pi) * (G.K @ (BOm_s * G.ys[:, None]) - tQ)
    ratio = torch.cat([Q_s[:1], BOm_s[1:] / G.ys[1:, None]])
    up0 = -(2 / math.pi) * (G.w @ ratio + tQ)
    s_c = (1 + G.y * G.y) ** (-alpha / 2)
    wt = G.y * (1 + G.y * G.y) ** (beta - 1) * (1 - s_c) ** (kappa - 1)
    return BOm, BOmy, AU, AUp, up0, wt


def residual(lam, c, nu, G, with_jac=False):
    n = c.shape[0]
    kappa = (nu + 1) / 2
    BOm, BOmy, AU, AUp, up0, wt = operators(lam, kappa, G, n)
    Om, Omy, U, Up = BOm @ c, BOmy @ c, AU @ c, AUp @ c
    V = (1 + lam) * G.y - U
    R = (Om + V * Omy - Om * Up) / wt
    u0 = up0 @ c
    r = torch.cat([R, (c.sum() + 1.0).reshape(1), (nu * (1 + lam - u0) - (u0 - 1)).reshape(1)])
    if not with_jac:
        return r
    Jc = (BOm + V[:, None] * BOmy - Omy[:, None] * AU - Up[:, None] * BOm - Om[:, None] * AUp) / wt[:, None]
    Jc = torch.cat([Jc, torch.ones(1, n, dtype=DT, device=c.device), (-(nu + 1) * up0)[None]])
    return r, Jc


def res_jac(p, nu, G, lam_free=True):
    lam, c = p[0], p[1:]
    r, Jc = residual(lam, c, nu, G, with_jac=True)
    if not lam_free:
        return r, Jc
    e = 1e-6
    dl = (residual(lam + e, c, nu, G) - residual(lam - e, c, nu, G)) / (2 * e)
    return r, torch.cat([dl[:, None], Jc], 1)


def lm(p0, nu, G, lam_free=True, max_iter=200, tol=1e-15, mu0=1e-3):
    p = p0.clone()
    r, J = res_jac(p, nu, G, lam_free)
    cost = 0.5 * (r @ r)
    mu, hist = mu0, []
    sl = slice(0, None) if lam_free else slice(1, None)
    for it in range(max_iter):
        d = J.norm(dim=0).clamp_min(1e-30)
        Js = J / d
        m = Js.shape[1]
        ok = False
        for _ in range(40):
            Qm, Rm = torch.linalg.qr(torch.cat([Js, math.sqrt(mu) * torch.eye(m, dtype=DT, device=p.device)]))
            step = torch.linalg.solve_triangular(Rm, (Qm.T @ torch.cat([-r, torch.zeros(m, dtype=DT, device=p.device)]))[:, None], upper=True)[:, 0] / d
            pn = p.clone(); pn[sl] = pn[sl] + step
            rn = residual(pn[0], pn[1:], nu, G)
            cn = 0.5 * (rn @ rn)
            if torch.isfinite(cn) and cn < cost:
                rel = ((cost - cn) / cost).item()
                p, cost, mu, ok = pn, cn, max(mu / 5, 1e-16), True
                break
            mu *= 4
        if not ok:
            break
        r, J = res_jac(p, nu, G, lam_free)
        hist.append((it, r.abs().max().item(), r.norm().item(), p[0].item()))
        if hist[-1][1] < tol or (rel < 1e-13 and mu < 1e-12) or step.abs().max().item() < 1e-16:
            break
    return p, hist
