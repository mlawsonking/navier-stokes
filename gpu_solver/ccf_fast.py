"""CCF profile solver with analytic Jacobian (same discretisation as ccf.py, k=1 family).

For fixed lam every field is linear in the coefficients c: Omega = B_Om c, U = A_U c, ... so the residual
is quadratic in c and its c-Jacobian is explicit. The lam-column comes from one forward-mode derivative.
Unknowns p = [lam, c_0..c_(n-1)]. Rows: collocation, Omega'(0) = -2, lam + 2 - 2 U'(0) = 0.
"""
import math
import torch
from torch.func import jvp
from .ccf import cheb3, Grid, DT


MAP = None   # None: plain Chebyshev in x = 2s-1.  (x0, eps): Tee-Trefethen sinh map concentrating points at x0,
             # for a profile with complex singularities near x0 +/- i eps. t(x) is analytic on [-1,1], so the
             # basis T_k(t(2s-1)) is still analytic in s at both ends (origin and far field unchanged).


FAR = None   # None, or (nf, Jf): append far-field blocks s^(2 j (1+lam)) P_j(s), j = 1..Jf, P_j plain Chebyshev
             # with nf terms. They carry the y^(-2j) corrections (from r = 1 + y^2 and from the moments of the
             # Hilbert transform), which are powers s^(2j(1+lam)) and so NOT analytic in s at the far field.


def basis3(s, n, lam=None):
    """phi_k(s), d phi_k/ds, d^2 phi_k/ds^2 for k < n (the last nf*Jf of them far-field blocks if FAR)."""
    if FAR is None or lam is None:
        return _mapped3(s, n)
    nf, Jf = FAR
    out = [_mapped3(s, n - nf * Jf)]
    T, d1, d2 = cheb3(2 * s - 1, nf); P, P1, P2 = T, 2 * d1, 4 * d2
    for j in range(1, Jf + 1):
        g = 2 * j * (1 + lam)
        f0, f1, f2 = (s ** g)[:, None], (g * s ** (g - 1))[:, None], (g * (g - 1) * s ** (g - 2))[:, None]
        out.append((f0 * P, f0 * P1 + f1 * P, f0 * P2 + 2 * f1 * P1 + f2 * P))
    return tuple(torch.cat([o[i] for o in out], 1) for i in range(3))


def _mapped3(s, n):
    x = 2 * s - 1
    if MAP is None:
        T, d1, d2 = cheb3(x, n)
        return T, 2 * d1, 4 * d2
    x0, eps = MAP
    a1, a2 = math.asinh((1 - x0) / eps), math.asinh((1 + x0) / eps)
    u = x - x0
    t = 2 * (torch.asinh(u / eps) + a2) / (a1 + a2) - 1
    rt = torch.sqrt(eps * eps + u * u)
    ts = 2 * (2 / (a1 + a2)) / rt                       # dt/ds
    tss = 4 * (-(2 / (a1 + a2)) * u / rt ** 3)          # d2t/ds2
    T, d1, d2 = cheb3(t, n)
    return T, d1 * ts[:, None], d2 * (ts * ts)[:, None] + d1 * tss[:, None]


def _basis(lam, y, n):
    alpha = 1.0 / (1.0 + lam); beta = (1 - alpha) / 2
    r = 1 + y * y
    s = r ** (-alpha / 2)
    T, M1, M2 = basis3(s, n, lam)
    Q = 2 * beta * T - alpha * s[:, None] * M1
    Q1 = (2 * beta - alpha) * M1 - alpha * s[:, None] * M2
    BTh = (r ** beta)[:, None] * T
    BOm = (y * r ** (beta - 1))[:, None] * Q
    BOmy = (r ** (beta - 1))[:, None] * Q + (y * y * r ** (beta - 2))[:, None] * (2 * (beta - 1) * Q - alpha * s[:, None] * Q1)
    return BTh, BOm, BOmy, Q, alpha, beta


def operators(lam, G, n):
    BTh_s, BOm_s, _, Q_s, alpha, beta = _basis(lam, G.ys, n)
    _, BOm, BOmy, _, _, _ = _basis(lam, G.y, n)
    sY = (1 + G.Y * G.Y) ** (-alpha / 2)
    sg = sY * G.g
    Tg, M1g, _ = basis3(sg, n, lam)
    tM = sY * (G.gw @ Tg) / alpha
    tQ = sY * (G.gw @ (2 * beta * Tg - alpha * sg[:, None] * M1g)) / alpha
    AU = (2 * G.y / math.pi)[:, None] * (G.K @ BTh_s - tM)
    AUp = (2 / math.pi) * (G.K @ (BOm_s * G.ys[:, None]) - tQ)
    ratio = torch.cat([Q_s[:1], BOm_s[1:] / G.ys[1:, None]])
    up0 = -(2 / math.pi) * (G.w @ ratio + tQ)
    q1 = Q_s[0]                                    # Omega'(0) functional (s = 1 at the first source, y = 0)
    wt = G.y * (1 + G.y * G.y) ** (beta - 1)
    return BOm, BOmy, AU, AUp, up0, q1, wt


MODE = dict(norm='origin', origin_row=True)   # experiment switches (module-level; default = rung 2a stable setup)


def residual(lam, c, G, a=-1.0, with_jac=False):
    n = c.shape[0]
    BOm, BOmy, AU, AUp, up0, q1, wt = operators(lam, G, n)
    Om, Omy, U, Up = BOm @ c, BOmy @ c, AU @ c, AUp @ c
    V = (1 + lam) * G.y + a * U
    R = (Om + V * Omy - Om * Up) / wt
    keep = G.y >= MODE.get('ymin', 0.0)              # optional: drop collocation rows near the origin
    R = R[keep]
    if MODE['norm'] == 'origin':
        nvec, nval = q1, MODE.get('d', 2.0)
    else:                                             # tail amplitude: M(0) = -1
        nvec = basis3(torch.zeros(1, dtype=DT, device=c.device), n, lam)[0][0]; nval = 1.0
    rows = [R, (nvec @ c + nval).reshape(1)]
    if MODE['origin_row']:
        rows.append((lam + 2 - (1 - a) * (up0 @ c)).reshape(1))
    r = torch.cat(rows)
    if not with_jac:
        return r
    Jc = ((BOm + V[:, None] * BOmy + a * Omy[:, None] * AU - Up[:, None] * BOm - Om[:, None] * AUp) / wt[:, None])[keep]
    Jc = torch.cat([Jc, nvec[None]] + ([-(1 - a) * up0[None]] if MODE['origin_row'] else []))
    return r, Jc


def res_jac(p, G, a=-1.0, lam_free=True):
    lam, c = p[0], p[1:]
    r, Jc = residual(lam, c, G, a, with_jac=True)
    if not lam_free:
        return r, Jc
    e = 1e-6      # central difference for the lam column (forward-mode through the recurrences is 10x slower)
    dl = (residual(lam + e, c, G, a) - residual(lam - e, c, G, a)) / (2 * e)
    return r, torch.cat([dl[:, None], Jc], 1)


def lm(p0, G, a=-1.0, lam_free=True, max_iter=200, tol=1e-15, mu0=1e-3, verbose=False, stall=0):
    """LM on p = [lam, c] (lam_free) or on c with p[0] fixed. Returns p, history [(it, sup, l2, lam)]."""
    p = p0.clone()
    r, J = res_jac(p, G, a, lam_free)
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
            rn = residual(pn[0], pn[1:], G, a)
            cn = 0.5 * (rn @ rn)
            if torch.isfinite(cn) and cn < cost:
                rel = ((cost - cn) / cost).item()
                p, cost, mu, ok = pn, cn, max(mu / 5, 1e-16), True
                break
            mu *= 4
        if not ok:
            break
        r, J = res_jac(p, G, a, lam_free)
        hist.append((it, r.abs().max().item(), r.norm().item(), p[0].item()))
        if verbose and it % 10 == 0:
            print(f"  it {it} sup {hist[-1][1]:.2e} lam {p[0].item():.12f} mu {mu:.1e}", flush=True)
        if hist[-1][1] < tol or (rel < 1e-13 and mu < 1e-12) or step.abs().max().item() < 1e-16:
            break
        if stall and len(hist) > stall and hist[-1][1] > 0.5 * hist[-1 - stall][1] and hist[-1][1] < 1e-6:
            break                                   # at the noise floor: sup no longer halving
    return p, hist
