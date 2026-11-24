"""The SINGULAR end profile of the non-smooth CCF family (a = -1, Burgers form): a cusp at a stagnation point.

    Theta' = Omega, Theta even, U = H Theta,   -lam Theta + V Theta' = 0,   V = (1+lam) y - U.

Local structure at y* (derived in runs/sing_end_local.py, see the report there): Phi = Theta + i U is the boundary
value of an analytic function; with t = |d|^(1/2), d = y - y*, the expansion Phi = sum_k phi_k zeta^(k/2) gives one
2x2 linear system per order k whose determinant vanishes iff
    tan(pi k/2) = (rho + 1/rho) k / (1 - k^2),      rho = p_1/q_1 (ratio of the two one-sided cusp amplitudes);
for rho = 1 the roots are k = 1 (amplitude ratio), 3.66040 (= 1 + 2 x 1.33020, the 'even' inner mode) and
5.78204 ('odd' mode), ...; all integer k > 1 are non-resonant. So on each side Theta is an analytic function of t
plus weak terms t^3.66.. (C^3 in t), which Chebyshev in t resolves at an algebraic rate n^-8.3.

Representation (no explicit cusp functions needed): with the usual s = r^(-alpha/2), r = 1 + y^2, s* = s(y*),
    Theta = r^beta (1-s)^(mu/2) N(s) + origin blocks of ccf_sing (exponents mu + j(mu-1), unchanged),
    N = sum_k cL_k T_k(2 sig - 1)  for y < y*,  sig^2 = (s - s*)/(1 - s*)      (sig = 0 at y*, 1 at y = 0)
    N = sum_k cR_k T_k(2 sig - 1)  for y > y*,  sig^2 = (s* - s)/s*            (sig = 0 at y*, 1 at y = inf)
plus one continuity row N_L(y*) = N_R(y*). y* is FIXED (it fixes the scaling y -> a y, Theta -> a Theta; the
standard normalisation is restored afterwards, as in sing_track4). Nothing imposes V(y*) = 0: it is forced by the
equation (Omega ~ |d|^-1/2) and is reported as a diagnostic.

Quadrature: on each side the nodes are uniform in an auxiliary variable tau with u(tau) analytic,
theta = pi / (1 + e^-u), sig = cos^2(theta/2): exponential clustering at the origin, at both sides of y* and at
infinity (every power |d|^g, |y|^g becomes an exponential in u), and in the middle the density follows the
Chebyshev density (R nodes per basis function). The Hilbert transform is computed by singularity subtraction,
    U(y) = (2y/pi) [ sum_j w_j (Theta_j - Theta(y)) / (y^2 - y_j^2) + Theta(y) I(y) - tail ],
so rows can sit anywhere (no half-step rule), and node differences y - y_j are formed from the accurate d's.
"""
import math
import numpy as np
import torch
from .ccf import cheb3, DT


KF = 3.660448369140012    # cusp exponent (in t = |d|^1/2) for A_R = A_L: root of tan(pi k/4) = -1/k


class Cusp:
    """Columns: [cL (nL) | cR (nR) | origin blocks (nj J) | cusp blocks L, R (nb each) | far block (nfar)].
    Cusp block: (1-s)^(mu/2) sig^kf T_k(2 sig - 1) on one side (the free t^3.66.. terms; kf = KF assumes A_R = A_L).
    Far block (right side): (1-s)^(mu/2) s^(2(1+lam)) T_k(2 sig - 1): the y^-2 far-field corrections, which are not
    analytic in s (ccf_fast.FAR).
    Tested 2026-11-23 (256/128, fixed lam): nb = 8 lowers the residual from 1.8e-10 to 5e-12, but ANY kf in 3.2..4.5
    does the same and lam, mu, y* agree to 1e-12 -- the block acts as extra resolution at the cusp; it does NOT verify
    the exponent 3.6604. nfar = 12 alone changes nothing at this level."""
    def __init__(self, ystar, nL, nR, nj, J, nb=0, nfar=0, kf=KF):
        self.yst, self.nL, self.nR, self.nj, self.J = float(ystar), nL, nR, nj, J
        self.nb, self.nfar, self.kf = nb, nfar, kf
        self.n0 = nL + nR + nj * J
        self.n = self.n0 + 2 * nb + nfar

    def geom(self, lam, y, d):
        """s, 1-s, sigma, 1-sigma, left-mask for points y (d = y - y* given accurately)."""
        al = 1.0 / (1.0 + lam); rst = 1 + self.yst ** 2; sst = rst ** (-al / 2)
        l = torch.log1p(y * y); s = torch.exp(-al / 2 * l); oms = -torch.expm1(-al / 2 * l)
        q = torch.log1p(d * (2 * self.yst + d) / rst)
        dss = sst * torch.expm1(-al / 2 * q)                      # s - s*
        left = d < 0
        sig2 = torch.where(left, dss / (1 - sst), -dss / sst)
        om2 = torch.where(left, oms / (1 - sst), s / sst)          # 1 - sig^2
        sig = torch.sqrt(sig2.clamp_min(0)); omsig = om2 / (1 + sig)
        dxds = torch.where(left, 1 / (sig * (1 - sst)), -1 / (sig * sst))
        return s, oms, l, sig, omsig, left, dxds, al

    def basis(self, lam, mu, y, d):
        """E (values of M = Theta / r^beta), E1 = dM/ds, and s, 1-s, log r."""
        s, oms, l, sig, omsig, left, dxds, al = self.geom(lam, y, d)
        th = 2 * torch.atan2(torch.sqrt(omsig), torch.sqrt(sig))    # x = 2 sig - 1 = cos th
        sth = 2 * torch.sqrt(sig * omsig)
        f = oms ** (mu / 2); fx = (mu / 2) * oms ** (mu / 2 - 1)    # d f / d(1-s)
        blocks, dblocks = [], []
        for n, msk in ((self.nL, left), (self.nR, ~left)):
            k = torch.arange(n, dtype=DT, device=y.device)
            T = torch.cos(k[None, :] * th[:, None])
            dT = k[None, :] * torch.sin(k[None, :] * th[:, None]) / sth[:, None] * dxds[:, None]   # dT/ds
            m = msk.to(DT)[:, None]
            blocks.append(m * f[:, None] * T); dblocks.append(m * (f[:, None] * dT - fx[:, None] * T))
        for j in range(1, self.J + 1):                              # origin blocks, as ccf_sing.E2
            g = (mu + j * (mu - 1)) / 2
            T, d1, _ = cheb3(2 * s - 1, self.nj); P, P1 = T, 2 * d1
            if j % 2 == 0:
                g0 = (2 + j) / 2; m2 = mu - 2; lx = torch.log(oms)
                dd = (j + 1) / 2 * lx if abs(float(m2)) < 1e-9 else torch.expm1((j + 1) / 2 * m2 * lx) / m2
                ff = oms ** g0 * dd; ffx = oms ** (g0 - 1) * (g * dd + (j + 1) / 2)
            else:
                ff = oms ** g; ffx = g * oms ** (g - 1)
            blocks.append(ff[:, None] * P); dblocks.append(ff[:, None] * P1 - ffx[:, None] * P)
        cheb = lambda n: (lambda k: (torch.cos(k[None, :] * th[:, None]),
                                     k[None, :] * torch.sin(k[None, :] * th[:, None]) / sth[:, None] * dxds[:, None]))(torch.arange(n, dtype=DT, device=y.device))
        if self.nb:
            T, dT = cheb(self.nb); g = sig ** self.kf; gs = self.kf * sig ** (self.kf - 1) * dxds / 2     # d sig/ds = dxds/2
            for msk in (left, ~left):
                m = msk.to(DT)[:, None]
                blocks.append(m * (f * g)[:, None] * T)
                dblocks.append(m * ((f * g)[:, None] * dT + (f * gs - fx * g)[:, None] * T))
        if self.nfar:
            T, dT = cheb(self.nfar); e = 2 / al; g = s ** e; gs = e * s ** (e - 1); m = (~left).to(DT)[:, None]
            blocks.append(m * (f * g)[:, None] * T)
            dblocks.append(m * ((f * g)[:, None] * dT + (f * gs - fx * g)[:, None] * T))
        return torch.cat(blocks, 1), torch.cat(dblocks, 1), s, oms, l, al

    def fields(self, lam, mu, y, d):
        E, E1, s, oms, l, al = self.basis(lam, mu, y, d); beta = (1 - al) / 2
        rb = torch.exp(beta * l)
        Q = 2 * beta * E - al * s[:, None] * E1
        return rb[:, None] * E, (y * rb / (1 + y * y))[:, None] * Q, oms, rb, al

    def y_of_s(self, s, al):
        y = torch.sqrt(torch.expm1(-(2 / al) * torch.log(s)))
        return y, y - self.yst

    def standard(self, p):
        """factor a with a Theta(y/a) ~ -(0.1/2)|y|^mu; y* in the standard normalisation is a y*."""
        lam, mu = p[0].item(), p[1].item(); al = 1 / (1 + lam)
        c = p[2:]; i = self.n0
        amp = -2 * (al / 2) ** (mu / 2) * (c[:self.nL].sum() + c[i:i + self.nb].sum()).item()     # all blocks with T_k(1) = 1 at y = 0
        return (amp / 0.1) ** (1 / (mu - 1))


def _u_of_tau(tau, h0, C):
    """invert tau(u) = u/h0 + 2 C tanh(u/2) (monotone)."""
    u = tau / (1 / h0 + C)
    for _ in range(200):
        t = torch.tanh(u / 2)
        u = u - (u / h0 + 2 * C * t - tau) / (1 / h0 + C * (1 - t * t))
    return u


class CuspGrid:
    """Nodes (sources) and rows on both sides of y*. Built with a reference lam (the nodes stay fixed in y while
    lam changes; the basis is evaluated with the actual lam). R_src: nodes per Chebyshev function in the middle of
    each side; h0: spacing in u far out; rows: m per source interval at tau = j + (i + off)/m (never on a node)."""

    def __init__(self, cusp, lam_ref, R_src=4.0, m=2, h0=0.05, off=0.5, row_lims=(-12.0, -14.0, 22.0),
                 src_lims=(-41.0, -41.0, 48.0), device="cuda", n_gl=128):
        self.c = cusp; al = 1 / (1 + lam_ref); self.al_ref = al
        yst = cusp.yst; rst = 1 + yst ** 2; sst = rst ** (-al / 2)
        ln0, lnc, lninf = src_lims; rn0, rnc, rninf = row_lims
        out = {}
        for side, n in (("L", cusp.nL), ("R", cusp.nR)):
            C = R_src * n / 4.0
            base = torch.arange(math.floor(-60 / h0 - 2 * C) - 2, math.ceil(30 / h0 + 2 * C) + 2, dtype=DT, device=device)
            for kind in ("src", "row"):
                if kind == "src":
                    tt = base
                else:
                    tt = (base[:, None] + (torch.arange(m, dtype=DT, device=device)[None, :] + off) / m).flatten()
                u = _u_of_tau(tt, h0, C)
                y, d, dydu = self._map(u, side, al, sst, rst, yst)
                dudt = 1 / (1 / h0 + C * (1 - torch.tanh(u / 2) ** 2))
                lo, hi = (ln0, lnc) if kind == "src" else (rn0, rnc)
                if side == "L":            # origin limit lo, cusp limit hi (logs of y and of |d|/y*)
                    keep = (torch.log(y) > lo) & (torch.log(-d / yst) > hi)
                else:                      # cusp limit hi, far limit top
                    top = lninf if kind == "src" else rninf
                    keep = (torch.log(d / yst) > hi) & (torch.log(y) < top)
                idx = keep.nonzero().flatten()
                w = (dydu * dudt)[idx]
                if kind == "src":
                    w = w.clone(); w[0] *= 0.5; w[-1] *= 0.5
                out[(side, kind)] = (y[idx], d[idx], w)
        cat = lambda kind, i: torch.cat([out[("L", kind)][i], out[("R", kind)][i]])
        self.ys, self.ds, self.w = cat("src", 0), cat("src", 1), cat("src", 2)
        self.y, self.d = cat("row", 0), cat("row", 1)
        self.Y = self.ys.max()
        g, gw = np.polynomial.legendre.leggauss(n_gl)
        self.g = torch.tensor((g + 1) / 2, dtype=DT, device=device); self.gw = torch.tensor(gw / 2, dtype=DT, device=device)
        # kernel with accurate differences: y_c - y_j = d_c - d_j
        dif = self.d[:, None] - self.ds[None, :]
        self.K = self.w[None, :] / (dif * (self.y[:, None] + self.ys[None, :]))
        Yv = self.Y
        self.I = torch.log((Yv + self.y) / (Yv - self.y)) / (2 * self.y)      # PV int_0^Y dy'/(y^2 - y'^2)

    @staticmethod
    def _map(u, side, al, sst, rst, yst):
        # theta = pi/(1+e^-u); sig = cos^2(theta/2) = sin^2((pi-theta)/2); 1 - sig = sin^2(theta/2)
        th = math.pi / (1 + torch.exp(-u)); pth = math.pi / (1 + torch.exp(u))
        sig = torch.sin(pth / 2) ** 2; omsig = torch.sin(th / 2) ** 2
        sth = 2 * torch.sqrt(sig * omsig); dthdu = th * pth / math.pi
        if side == "L":
            dss = (1 - sst) * sig * sig; oms = (1 - sst) * omsig * (1 + sig); s = 1 - oms
            logs = torch.log1p(-oms); dsdsig = 2 * (1 - sst) * sig
        else:
            dss = -sst * sig * sig; s = sst * omsig * (1 + sig)
            logs = torch.log(s); dsdsig = -2 * sst * sig
        y = torch.sqrt(torch.expm1(-(2 / al) * logs)); r = 1 + y * y
        d = rst * torch.expm1(-(2 / al) * torch.log1p(dss / sst)) / (y + yst)
        dyds = -r / (al * y * s)
        dydu = (dyds * dsdsig * (-sth / 2) * dthdu).abs()
        return y, d, dydu


def operators(cusp, lam, mu, G):
    BTh_s, _, _, _, al = cusp.fields(lam, mu, G.ys, G.ds)
    BTh, BOm, oms, rb, _ = cusp.fields(lam, mu, G.y, G.d)
    sY = (1 + G.Y * G.Y) ** (-al / 2); sg = sY * G.g
    yg, dg = cusp.y_of_s(sg, al)
    Eg = cusp.basis(lam, mu, yg, dg)[0]
    tM = sY * (G.gw @ Eg) / al                                    # int_Y^inf Theta / y'^2 dy'
    AU = (2 * G.y / math.pi)[:, None] * (G.K @ BTh_s - (G.K.sum(1) - G.I)[:, None] * BTh - tM[None, :])
    up0 = -(2 / math.pi) * (G.w @ (BTh_s / (G.ys ** 2)[:, None]) + tM)
    wt = oms ** (mu / 2) * rb
    nL, nR = cusp.nL, cusp.nR
    cont = torch.zeros(cusp.n, dtype=DT, device=G.y.device)
    sg_ = torch.tensor([(-1.0) ** k for k in range(max(nL, nR))], dtype=DT, device=G.y.device)
    # (the cusp blocks vanish at y*; the far block does not)
    cont[:nL] = sg_[:nL]; cont[nL:nL + nR] = -sg_[:nR]
    if cusp.nfar:
        sst = (1 + cusp.yst ** 2) ** (-al / 2); k = torch.arange(cusp.nfar, dtype=DT, device=G.y.device)
        cont[cusp.n0 + 2 * cusp.nb:] = -sst ** (2 / al) * (-1.0) ** k
    return BTh, BOm, AU, up0, wt, cont


WC = 10.0      # weight of the continuity row
RHO = None     # None, or the imposed ratio A_R/A_L of the one-sided cusp amplitudes (Omega ~ -A |d|^-1/2): one extra row
WR = 10.0


def amp_rows(cusp, lam):
    """Linear forms aL, aR with A_L = (aL . c) and A_R = (aR . c) up to the common factor r*^beta (1-s*)^(mu/2) sqrt|s'(y*)|/2:
    Theta = Theta* + F sig + ..., A_L = F_L / (2 sqrt(1-s*)) .., A_R = -F_R / (2 sqrt s*) .., dT_k/dsig(0) = 2 (-1)^(k+1) k^2."""
    al = 1 / (1 + lam); sst = (1 + cusp.yst ** 2) ** (-al / 2)
    aL = torch.zeros(cusp.n, dtype=DT, device="cuda"); aR = torch.zeros_like(aL)
    k = torch.arange(max(cusp.nL, cusp.nR), dtype=DT, device="cuda"); dT = 2 * (-1) ** (k + 1) * k * k
    aL[:cusp.nL] = dT[:cusp.nL] / (2 * math.sqrt(1 - sst)); aR[cusp.nL:cusp.nL + cusp.nR] = -dT[:cusp.nR] / (2 * math.sqrt(sst))
    if cusp.nfar:
        kf_ = torch.arange(cusp.nfar, dtype=DT, device="cuda")
        aR[cusp.n0 + 2 * cusp.nb:] = -sst ** (2 / al) * 2 * (-1) ** (kf_ + 1) * kf_ * kf_ / (2 * math.sqrt(sst))
    return aL, aR


def residual(cusp, p, G, with_jac=False):
    lam, mu, c = p[0], p[1], p[2:]
    BTh, BOm, AU, up0, wt, cont = operators(cusp, lam.item(), mu.item(), G)
    Th, Om, U = BTh @ c, BOm @ c, AU @ c
    V = (1 + lam) * G.y - U
    R = (-lam * Th + V * Om) / wt
    u0 = up0 @ c
    extra, Jx = [], []
    if RHO is not None:               # A_R - RHO A_L = 0, scaled by A_L (placed before the last two rows)
        aL, aR = amp_rows(cusp, lam.item()); nrm = (aL @ c).abs()
        extra = [(WR * (aR @ c - RHO * (aL @ c)) / nrm).reshape(1)]; Jx = [(WR * (aR - RHO * aL) / nrm)[None]]
    r = torch.cat([R] + extra + [(WC * (cont @ c)).reshape(1), (-lam + (1 + lam - u0) * mu).reshape(1)])
    if not with_jac:
        return r
    Jc = (-lam * BTh + V[:, None] * BOm - Om[:, None] * AU) / wt[:, None]
    return r, torch.cat([Jc] + Jx + [WC * cont[None], (-mu * up0)[None]])


def res_jac(cusp, p, G, step=2e-7, free=(0, 1)):
    r, Jc = residual(cusp, p, G, with_jac=True)
    cols = []
    for k in free:
        e = torch.zeros_like(p); e[k] = step
        f = lambda q: residual(cusp, q, G)
        cols.append(((8 * (f(p + e) - f(p - e)) - (f(p + 2 * e) - f(p - 2 * e))) / (12 * step))[:, None])
    return r, torch.cat(cols + [Jc], 1)


def UV_at(cusp, p, G, y, d):
    """U and V at arbitrary points (y, d) with the quadrature of G (sources only)."""
    lam, mu, c = p[0].item(), p[1].item(), p[2:]
    BTh_s, _, _, _, al = cusp.fields(lam, mu, G.ys, G.ds)
    BTh = cusp.fields(lam, mu, y, d)[0]
    sY = (1 + G.Y * G.Y) ** (-al / 2); sg = sY * G.g; yg, dg = cusp.y_of_s(sg, al)
    tM = sY * (G.gw @ cusp.basis(lam, mu, yg, dg)[0]) / al
    K = G.w[None, :] / ((d[:, None] - G.ds[None, :]) * (y[:, None] + G.ys[None, :]))
    I = torch.log((G.Y + y) / (G.Y - y)) / (2 * y)
    Th_s = BTh_s @ c; Th = BTh @ c
    U = (2 * y / math.pi) * (K @ Th_s - (K.sum(1) - I) * Th - tM @ c)
    return U, (1 + lam) * y - U, Th
