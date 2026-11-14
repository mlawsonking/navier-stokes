"""CCF (a = -1) profiles that are NOT smooth at the origin: the one-parameter family in lam.

Burgers form (Theta' = Omega, Theta(0) = 0, U = H Theta):
    -lam Theta + ((1+lam) y - U) Theta' = 0.
Local analysis at y = 0 (derived here, unchecked elsewhere): Theta ~ A |y|^mu with
    mu = lam / (1 + lam - U'(0)),
and because H(|y|^g) has the non-analytic part ~ sgn(y)|y|^g, the corrections carry the exponents
    mu + j (mu - 1) + 2k,   j, k >= 0.
mu = 2 is the smooth case (the j >= 1 terms vanish). Representation, with s = r^(-alpha/2), r = 1 + y^2,
1 - s ~ (alpha/2) y^2 at the origin and analytic at the far field:
    Theta = r^beta (1-s)^(mu/2) [ N_0(s) + sum_{j=1..J} (1-s)^(j (mu-1)/2) N_j(s) ]
N_0: mapped Chebyshev series (ccf_fast.MAP), n0 terms. N_j: plain Chebyshev, nj terms.
Unknowns p = [lam, mu, c]. Rows: collocation of the Burgers form on y = exp(xi) (so |y|^g is analytic in xi
and the trapezoid rule keeps spectral accuracy), amplitude  Theta ~ -(d/2)|y|^mu,  origin relation.
`free` chooses the problem: (False, True) family at fixed lam; (True, False) smooth profile with lam unknown.
"""
import math
import numpy as np
import torch
from .ccf import cheb3, DT
from . import ccf_fast as F


class EGrid:
    """Sources y' = exp(xi(tau_j)), tau_j = j h; collocation at the half-steps. cluster = (xi_p, w, A) uses
    xi(tau) = tau - A atan((tau - tau_p)/w): analytic, and denser by 1/(1 - A/w) around xi_p (the peak), so the
    rows keep up with a basis that is concentrated there. cluster = None is the plain exponential grid."""
    def __init__(self, h=0.0125, L=40.0, Lc0=12.0, Lc=22.0, n_gl=128, device="cuda", cluster=None):
        self.h = h
        if cluster is not None and cluster[0] == "sinh":
            # Row density that follows the sinh-mapped basis: tau(xi) = xi + D e asinh((xi - xp)/e), so
            # d tau/d xi = 1 + D e / sqrt(e^2 + (xi - xp)^2): D+1 times denser at the peak, falling off like the
            # basis density does. Inverted by Newton (monotone). Sources tau_a + j h, collocation at half steps.
            _, xp, e, D = cluster
            tau_of = lambda x: x + D * e * torch.asinh((x - xp) / e)
            T = lambda v: tau_of(torch.tensor(v, dtype=DT, device=device)).item()
            ta = T(-L); nJ = int(round((T(L) - ta) / h))
            i0 = int(math.ceil((T(-Lc0) - ta) / h - 0.5)); i1 = int(math.floor((T(Lc) - ta) / h - 0.5))
            def inv(tau):
                x = torch.where(tau < T(xp), tau - T(xp) + xp, tau - T(xp) + xp)
                x = xp + e * torch.sinh(((tau - T(xp)) / (D * e + 1e-300)).clamp(-30, 30)) if D > 0 else x
                x = x.clamp(-L - 1, L + 1)
                for _ in range(80):
                    x = x - (tau_of(x) - tau) / (1 + D * e / torch.sqrt(e * e + (x - xp) ** 2))
                return x
            xs = inv(ta + h * torch.arange(nJ + 1, dtype=DT, device=device))
            xc = inv(ta + h * (torch.arange(i0, i1 + 1, dtype=DT, device=device) + 0.5))
            self.ys = torch.exp(xs); self.w = h * self.ys / (1 + D * e / torch.sqrt(e * e + (xs - xp) ** 2))
            self.y = torch.exp(xc)
        else:
            if cluster is None:
                xi = lambda t: t; dxi = lambda t: torch.ones_like(t)
            elif cluster[0] == "box":
                # flat-top: density D for |xi - xp| < c (smooth edges), xi(tau) = tau - A B(tau - xp),
                # B' = [atan((u+b)/w) - atan((u-b)/w)]/pi, b = c D, w = b/5. Analytic; singularities at +-b +- i w.
                _, xp, c, D = cluster
                b = c * D; w = b / 5; A = (1 - 1 / D) / (2 * math.atan(5.0) / math.pi)
                Fp = lambda u: u * torch.atan(u / w) - (w / 2) * torch.log(w * w + u * u)
                xi = lambda t: t - A * (Fp(t - xp + b) - Fp(t - xp - b)) / math.pi
                dxi = lambda t: 1 - A * (torch.atan((t - xp + b) / w) - torch.atan((t - xp - b) / w)) / math.pi
                L, Lc0, Lc = L + A * b, Lc0 + A * b, Lc + A * b      # the map compresses the far ranges by A b
            else:
                xp, w, A = cluster
                xi = lambda t: t - A * torch.atan((t - xp) / w)
                dxi = lambda t: 1 - (A / w) / (1 + ((t - xp) / w) ** 2)
            J = int(round(L / h))
            tau = h * torch.arange(-J, J + 1, dtype=DT, device=device)
            self.ys = torch.exp(xi(tau))
            self.w = h * dxi(tau) * self.ys
            i = torch.arange(-int(round(Lc0 / h)), int(round(Lc / h)), dtype=DT, device=device)
            self.y = torch.exp(xi(h * (i + 0.5)))
        self.K = self.w[None, :] / (self.y[:, None] ** 2 - self.ys[None, :] ** 2)
        self.Y = self.ys[-1]
        g, gw = np.polynomial.legendre.leggauss(n_gl)
        self.g = torch.tensor((g + 1) / 2, dtype=DT, device=device)
        self.gw = torch.tensor(gw / 2, dtype=DT, device=device)


def E2(s, oms, mu, shape):
    """Basis for M(s) = Theta / r^beta and dM/ds. shape = (n0, nj, J). oms = 1 - s (passed in for accuracy)."""
    n0, nj, J = shape
    Es, E1s = [], []
    for j in range(J + 1):
        g = (mu + j * (mu - 1)) / 2
        if j == 0:
            P, P1, _ = F.basis3(s, n0)
        else:
            T, d1, _ = cheb3(2 * s - 1, nj); P, P1 = T, 2 * d1
        if j > 0 and j % 2 == 0:
            # even j: at mu = 2 the plain block (1-s)^g is an integer power, i.e. already in the span of N_0.
            # Use the divided difference [(1-s)^g - (1-s)^g0]/(mu-2), g0 = (2+j)/2, which tends to
            # ((j+1)/2) (1-s)^g0 log(1-s): independent of N_0 for every mu, so the Jacobian stays regular.
            g0 = (2 + j) / 2; m2 = mu - 2; lx = torch.log(oms)
            if abs(float(m2)) < 1e-9:
                dd = (j + 1) / 2 * lx
            else:
                dd = torch.expm1((j + 1) / 2 * m2 * lx) / m2
            f = oms ** g0 * dd
            fx = oms ** (g0 - 1) * (g * dd + (j + 1) / 2)          # d f / d(1-s)
        else:
            f = oms ** g
            fx = g * oms ** (g - 1)
        Es.append(f[:, None] * P)
        E1s.append(f[:, None] * P1 - fx[:, None] * P)
    return torch.cat(Es, 1), torch.cat(E1s, 1)


def _fields(lam, mu, y, shape):
    alpha = 1.0 / (1.0 + lam); beta = (1 - alpha) / 2
    l = torch.log1p(y * y)
    s = torch.exp(-alpha / 2 * l); oms = -torch.expm1(-alpha / 2 * l)
    E, E1 = E2(s, oms, mu, shape)
    Q = 2 * beta * E - alpha * s[:, None] * E1
    rb = torch.exp(beta * l)
    BTh = rb[:, None] * E
    BOm = (y * rb / (1 + y * y))[:, None] * Q
    return BTh, BOm, Q, oms, rb, alpha, beta


def operators(lam, mu, G, shape):
    BTh_s, BOm_s, _, _, _, alpha, beta = _fields(lam, mu, G.ys, shape)
    BTh, BOm, _, oms, rb, _, _ = _fields(lam, mu, G.y, shape)
    sY = (1 + G.Y * G.Y) ** (-alpha / 2)
    sg = sY * G.g
    Eg, E1g = E2(sg, 1 - sg, mu, shape)
    tM = sY * (G.gw @ Eg) / alpha
    tQ = sY * (G.gw @ (2 * beta * Eg - alpha * sg[:, None] * E1g)) / alpha
    AU = (2 * G.y / math.pi)[:, None] * (G.K @ BTh_s - tM)
    up0 = -(2 / math.pi) * (G.w @ (BOm_s / G.ys[:, None]) + tQ)
    wt = oms ** (mu / 2) * rb
    one = torch.ones(1, dtype=DT, device=G.y.device)
    nvec = torch.zeros(BTh.shape[1], dtype=DT, device=G.y.device)
    nvec[:shape[0]] = (alpha / 2) ** (mu / 2) * F.basis3(one, shape[0])[0][0]
    return BTh, BOm, AU, up0, wt, nvec


D = 0.1


def residual(p, G, shape, with_jac=False):
    lam, mu, c = p[0], p[1], p[2:]
    BTh, BOm, AU, up0, wt, nvec = operators(lam, mu, G, shape)
    Th, Om, U = BTh @ c, BOm @ c, AU @ c
    V = (1 + lam) * G.y - U
    R = (-lam * Th + V * Om) / wt
    u0 = up0 @ c
    r = torch.cat([R, (nvec @ c + D / 2).reshape(1), (-lam + (1 + lam - u0) * mu).reshape(1)])
    if not with_jac:
        return r
    Jc = (-lam * BTh + V[:, None] * BOm - Om[:, None] * AU) / wt[:, None]
    return r, torch.cat([Jc, nvec[None], (-mu * up0)[None]])


def res_jac(p, G, shape, free):
    r, Jc = residual(p, G, shape, with_jac=True)
    cols = []
    for k in (0, 1):
        if free[k]:
            e = torch.zeros_like(p); e[k] = 1e-6
            cols.append(((residual(p + e, G, shape) - residual(p - e, G, shape)) / 2e-6)[:, None])
    return r, torch.cat(cols + [Jc], 1)


def lm(p0, G, shape, free=(False, True), max_iter=200, tol=1e-14, mu0=1e-6, verbose=False, stall=5):
    p = p0.clone()
    idx = torch.tensor([k for k in (0, 1) if free[k]] + list(range(2, p.numel())), device=p.device)
    r, J = res_jac(p, G, shape, free)
    cost = 0.5 * (r @ r); damp, hist = mu0, []
    for it in range(max_iter):
        d = J.norm(dim=0).clamp_min(1e-30); Js = J / d; m = Js.shape[1]; ok = False
        eye = torch.eye(m, dtype=DT, device=p.device); z = torch.zeros(m, dtype=DT, device=p.device)
        for _ in range(40):
            Qm, Rm = torch.linalg.qr(torch.cat([Js, math.sqrt(damp) * eye]))
            step = torch.linalg.solve_triangular(Rm, (Qm.T @ torch.cat([-r, z]))[:, None], upper=True)[:, 0] / d
            pn = p.clone(); pn[idx] = pn[idx] + step
            rn = residual(pn, G, shape); cn = 0.5 * (rn @ rn)
            if torch.isfinite(cn) and cn < cost:
                rel = ((cost - cn) / cost).item()
                p, cost, damp, ok = pn, cn, max(damp / 5, 1e-16), True
                break
            damp *= 4
        if not ok:
            break
        r, J = res_jac(p, G, shape, free)
        hist.append((it, r.abs().max().item(), p[0].item(), p[1].item()))
        if verbose and it % 10 == 0:
            print(f"  it {it} sup {hist[-1][1]:.2e} lam {p[0].item():.12f} mu {p[1].item():.10f} damp {damp:.1e}", flush=True)
        if hist[-1][1] < tol or (rel < 1e-13 and damp < 1e-12) or step.abs().max().item() < 1e-16:
            break
        if len(hist) > stall and hist[-1][1] > 0.5 * hist[-1 - stall][1] and hist[-1][1] < 1e-8:
            break                                   # converged to the noise floor: sup no longer halving
    return p, hist


def from_smooth(p_old, old_map, new_map, shape, device):
    """Smooth solution in the ccf_fast representation (Theta = r^beta M(s), M(1) = 0) -> this one (mu = 2)."""
    n0 = shape[0]
    j = torch.arange(n0, dtype=DT, device=device); t = torch.cos(math.pi * (j + 0.5) / n0)
    if new_map is None:
        s = (t + 1) / 2
    else:
        x0, eps = new_map; b1, b2 = math.asinh((1 - x0) / eps), math.asinh((1 + x0) / eps)
        s = (x0 + eps * torch.sinh((b1 + b2) * (t + 1) / 2 - b2) + 1) / 2
    F.MAP = old_map
    M = F.basis3(s, p_old.numel() - 1)[0] @ p_old[1:].to(device)
    M1 = F.basis3(torch.ones(1, dtype=DT, device=device), p_old.numel() - 1)[0][0] @ p_old[1:].to(device)
    F.MAP = new_map
    c0 = torch.linalg.solve(F.basis3(s, n0)[0], (M - M1) / (1 - s))
    two = torch.tensor([2.0], dtype=DT, device=device)
    return torch.cat([p_old[:1].to(device), two, c0, torch.zeros(shape[1] * shape[2], dtype=DT, device=device)]), M1.item()


def remap0(p, old_map, new_map, shape, device):
    """Carry the N_0 block to another sinh map by Chebyshev interpolation."""
    n0 = shape[0]
    j = torch.arange(n0, dtype=DT, device=device); t = torch.cos(math.pi * (j + 0.5) / n0)
    x0, eps = new_map; b1, b2 = math.asinh((1 - x0) / eps), math.asinh((1 + x0) / eps)
    s = (x0 + eps * torch.sinh((b1 + b2) * (t + 1) / 2 - b2) + 1) / 2
    F.MAP = old_map; N = F.basis3(s, n0)[0] @ p[2:2 + n0]
    F.MAP = new_map; q = p.clone(); q[2:2 + n0] = torch.linalg.solve(F.basis3(s, n0)[0], N)
    return q


def tsvd_gn(resjac, resid, p0, idx, max_iter=25, rc=1e-9, tol=1e-14, verbose=False, svd=True):
    """Gauss-Newton with a truncated-SVD pseudo-inverse of the column-scaled Jacobian and step halving.
    The singular blocks make the Jacobian numerically rank-deficient (about 30 singular values below 1e-12
    near mu = 2); damping (LM) then crawls, while discarding those directions converges quadratically.
    resjac(p) -> (r, J over the unknowns idx), resid(p) -> r. Returns p, history [(it, sup, l2, t, rank)]."""
    p = p0.clone(); hist = []
    r, J = resjac(p)
    for it in range(max_iter):
        d = J.norm(dim=0).clamp_min(1e-300)
        if svd:
            # SVD on the CPU (LAPACK gesdd): the CUDA SVD is slow here and once hung. QR-Tikhonov with the
            # same cutoff does NOT converge (tested), so the truncation has to be a real SVD.
            U, sv, Vh = torch.linalg.svd((J / d).cpu(), full_matrices=False); keep = sv > rc * sv[0]
            st = (Vh[keep].T @ ((U[:, keep].T @ -r.cpu()) / sv[keep])).to(J.device) / d; rank = int(keep.sum())
        else:                                   # same filter as a QR-Tikhonov solve (GPU SVD is very slow here)
            m = J.shape[1]; eye = torch.eye(m, dtype=DT, device=J.device)
            Qm, Rm = torch.linalg.qr(torch.cat([J / d, (rc * 5.0) * eye]))
            st = torch.linalg.solve_triangular(Rm, (Qm[:J.shape[0]].T @ -r)[:, None], upper=True)[:, 0] / d; rank = m
        t, ok = 1.0, False
        while t > 1e-3:
            pn = p.clone(); pn[idx] = pn[idx] + t * st; rn = resid(pn)
            if torch.isfinite(rn).all() and rn.norm() < r.norm():
                ok = True; break
            t /= 2
        if not ok:
            break
        gain = 1 - (rn.norm() / r.norm()).item()
        p = pn; r, J = resjac(p)
        hist.append((it, r.abs().max().item(), r.norm().item(), t, rank))
        if verbose:
            print(f"  gn {it} sup {hist[-1][1]:.2e} t {t:.3f} rank {hist[-1][4]}", flush=True)
        if hist[-1][1] < tol or gain < 0.3 and hist[-1][1] < 1e-7:
            break
    return p, hist


def gn(p0, G, shape, free=(False, True), **kw):
    idx = torch.tensor([k for k in (0, 1) if free[k]] + list(range(2, p0.numel())), device=p0.device)
    return tsvd_gn(lambda p: res_jac(p, G, shape, free), lambda p: residual(p, G, shape), p0, idx, **kw)


def peak_grid(lam, ypk, mp, n, h=0.0125, R=6.0, device="cuda", shift=0.0, widen=1.0, **kw):
    """EGrid whose row density follows a sinh-mapped basis (map mp = (x0, eps) in x = 2s-1, n functions) centred
    on a peak at y = ypk, with at least R rows per basis function near the peak."""
    al = 1.0 / (1.0 + float(lam)); x0, eps = mp
    s = (1 + ypk * ypk) ** (-al / 2); dsdxi = al * s * ypk * ypk / (1 + ypk * ypk)
    e = eps / (2 * dsdxi)                                    # map width in xi = log y
    a12 = math.asinh((1 - x0) / eps) + math.asinh((1 + x0) / eps)
    D = R * h * n * 2 / (a12 * math.pi * e)
    return EGrid(h=h, device=device, cluster=("sinh", math.log(ypk) + shift * e, widen * e, D / widen), **kw)
