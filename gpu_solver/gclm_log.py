"""gCLM profile solver that is regular through lam = 0 (alpha = 1, beta = 0).

ccf_fast writes Theta = r^beta M(s). As lam -> 0 the far field Omega ~ q0 y^(-alpha) integrates to
q0 (r^beta - 1)/(2 beta) -> (q0/2) log r, so M ~ q0/(2 beta) blows up and at lam = 0 Theta has a logarithm.
Here      Theta = r^beta s N(s) + q0 (r^beta - 1)/(2 beta),
N a (mapped) Chebyshev series, q0 = far-field amplitude of Omega y^alpha. Every column is regular in beta
(the divided difference is evaluated with expm1). The additive constant of Theta does not matter: only
Omega, U' = H Omega and U = H Theta (H of a constant is zero) enter.
Unknowns p = [lam, q0, c_0..c_(n-1)]; "z" below is [q0, c]. Same rows and API as ccf_fast.
"""
import math
import torch
from .ccf import DT
from . import ccf_fast as F


def _dd(beta, x):
    """(exp(beta x) - 1)/beta, regular at beta = 0."""
    b = float(beta)
    return x if abs(b) < 1e-13 else torch.expm1(beta * x) / beta


def _basis(lam, y, n):
    alpha = 1.0 / (1.0 + lam); beta = (1 - alpha) / 2
    lr = torch.log1p(y * y); r = 1 + y * y
    s = torch.exp(-alpha / 2 * lr)
    T, T1, T2 = F.basis3(s, n)
    sc = s[:, None]
    P, P1, P2 = sc * T, T + sc * T1, 2 * T1 + sc * T2
    Q = 2 * beta * P - alpha * sc * P1
    Q1 = (2 * beta - alpha) * P1 - alpha * sc * P2
    rb = torch.exp(beta * lr)
    BTh = rb[:, None] * P
    BOm = (y * rb / r)[:, None] * Q
    BOmy = (rb / r)[:, None] * Q + (y * y * rb / r ** 2)[:, None] * (2 * (beta - 1) * Q - alpha * sc * Q1)
    one = torch.ones_like(y)
    cat = lambda v, B: torch.cat([v[:, None], B], 1)
    return (cat(_dd(beta, lr) / 2, BTh), cat(y * rb / r, BOm), cat(rb / r + 2 * (beta - 1) * y * y * rb / r ** 2, BOmy),
            cat(one, Q), alpha, beta, s)


def operators(lam, G, n):
    BTh_s, BOm_s, _, Q_s, alpha, beta, _ = _basis(lam, G.ys, n)
    _, BOm, BOmy, _, _, _, _ = _basis(lam, G.y, n)
    lY = math.log(G.Y.item())
    sY = math.exp(-alpha * lY) if not torch.is_tensor(alpha) else torch.exp(-alpha * lY)
    sg = sY * G.g
    Tg, T1g, _ = F.basis3(sg, n)
    Pg, P1g = sg[:, None] * Tg, Tg + sg[:, None] * T1g
    tM = torch.cat([((_dd(2 * beta, torch.tensor(lY, dtype=DT, device=G.y.device)) + 1) / ((1 - 2 * beta) * G.Y)).reshape(1), sY * (G.gw @ Pg) / alpha])
    tQ = torch.cat([(sY / alpha * torch.ones(1, dtype=DT, device=G.y.device)).reshape(1), sY * (G.gw @ (2 * beta * Pg - alpha * sg[:, None] * P1g)) / alpha])
    AU = (2 * G.y / math.pi)[:, None] * (G.K @ BTh_s - tM)
    AUp = (2 / math.pi) * (G.K @ (BOm_s * G.ys[:, None]) - tQ)
    ratio = torch.cat([Q_s[:1], BOm_s[1:] / G.ys[1:, None]])
    up0 = -(2 / math.pi) * (G.w @ ratio + tQ)
    q1 = Q_s[0]
    wt = G.y * torch.exp((beta - 1) * torch.log1p(G.y * G.y))
    return BOm, BOmy, AU, AUp, up0, q1, wt


D = 0.1


def residual(lam, z, G, a=-1.0, with_jac=False):
    n = z.shape[0] - 1
    BOm, BOmy, AU, AUp, up0, q1, wt = operators(lam, G, n)
    Om, Omy, U, Up = BOm @ z, BOmy @ z, AU @ z, AUp @ z
    V = (1 + lam) * G.y + a * U
    R = (Om + V * Omy - Om * Up) / wt
    r = torch.cat([R, (q1 @ z + D).reshape(1), (lam + 2 - (1 - a) * (up0 @ z)).reshape(1)])
    if not with_jac:
        return r
    Jz = (BOm + V[:, None] * BOmy + a * Omy[:, None] * AU - Up[:, None] * BOm - Om[:, None] * AUp) / wt[:, None]
    return r, torch.cat([Jz, q1[None], -(1 - a) * up0[None]])


def res_jac(p, G, a=-1.0):
    lam, z = p[0], p[1:]
    r, Jz = residual(lam, z, G, a, with_jac=True)
    e = 1e-6
    dl = (residual(lam + e, z, G, a) - residual(lam - e, z, G, a)) / (2 * e)
    return r, torch.cat([dl[:, None], Jz], 1)


def gn(p0, G, a=-1.0, **kw):
    from .ccf_sing import tsvd_gn
    return tsvd_gn(lambda p: res_jac(p, G, a), lambda p: residual(p[0], p[1:], G, a), p0, torch.arange(p0.numel(), device=p0.device), **kw)


def from_fast(p_old, n, nodes_s):
    """ccf_fast solution [lam, c] (same F.MAP) -> [lam, q0, c_N]:  M = s N + q0/(2 beta), q0 = 2 beta M(0)."""
    lam = p_old[0]; beta = lam / (2 * (1 + lam))
    zero = torch.zeros(1, dtype=DT, device=p_old.device)
    M0 = F.basis3(zero, p_old.numel() - 1)[0][0] @ p_old[1:]
    M = F.basis3(nodes_s, p_old.numel() - 1)[0] @ p_old[1:]
    cN = torch.linalg.solve(F.basis3(nodes_s, n)[0], (M - M0) / nodes_s)
    return torch.cat([lam.reshape(1), (2 * beta * M0).reshape(1), cN])
