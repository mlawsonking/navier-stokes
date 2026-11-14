"""Rung 2a: self-similar profiles of the generalised CLM family (a = -1 is Cordoba-Cordoba-Fontelos)

    omega_t + a u omega_x = u_x omega,  u_x = H omega,  H f(y) = (1/pi) p.v. int f(y')/(y-y') dy'
    omega = (T-t)^-1 Omega(y),  y = x/(T-t)^(1+lam)
    profile:  Omega + ((1+lam) y + a U) Omega' - Omega U' = 0,   U' = H Omega,  U(0) = 0.

Check of conventions: a = 0, lam = 0, Omega = -2y/(1+y^2), U' = 2/(1+y^2) is the exact CLM profile.
At the origin the equation gives  lam = (1-a) U'(0) - 2  (first Taylor order; derived here, see HANDOFF).

Representation. Work with the primitive Theta (Theta' = Omega, even): U = H Theta, U' = H Theta'.
    alpha = 1/(1+lam),  beta = (1-alpha)/2,  r = 1+y^2,  s = r^(-alpha/2) in (0,1]
    Theta = r^beta M(s),   M(s) = sum_k c_k T_k(2s-1)
so Theta is even and analytic at 0 and Omega ~ y^(-alpha) at infinity with corrections in powers of
y^(-alpha) (assumed form of the tail; convergence of c_k is the test).
    Omega  = y r^(beta-1) Q,                 Q  = 2 beta M - alpha s M'
    Omega' = r^(beta-1) Q + y^2 r^(beta-2) (2(beta-1) Q - alpha s Q'),   Q' = (2 beta - alpha) M' - alpha s M''
Normalisation Omega'(0) = Q(1) = -2.

Nonlocal terms by quadrature on y' = sinh(xi'), xi' = j h, j = 0..J (half line, parity used explicitly),
evaluated at the midpoints xi = (i+1/2) h. A simple pole midway between trapezoid nodes gives the principal
value with exponentially small error. The tail y' > Y = sinh(J h) is added in closed form:
    int_Y^inf y'^(-1-alpha) F(s') dy' = (1/alpha) int_0^{s_Y} F(s) ds     (Gauss-Legendre).
"""
import math
import numpy as np
import torch

DT = torch.float64


def cheb3(x, n):
    """T_k, T_k', T_k'' (k<n) by recurrence. x: [m] -> three [m,n] tensors."""
    T = [torch.ones_like(x), x]
    d1 = [torch.zeros_like(x), torch.ones_like(x)]
    d2 = [torch.zeros_like(x), torch.zeros_like(x)]
    for k in range(1, n - 1):
        T.append(2 * x * T[k] - T[k - 1])
        d1.append(2 * T[k] + 2 * x * d1[k] - d1[k - 1])
        d2.append(4 * d1[k] + 2 * x * d2[k] - d2[k - 1])
    return torch.stack(T[:n], -1), torch.stack(d1[:n], -1), torch.stack(d2[:n], -1)


class Grid:
    def __init__(self, h=0.1, L=40.0, Lc=22.0, n_gl=64, device="cuda", cluster=None):
        """cluster = (xi_p, w, A): xi(tau) = tau - A [atan((tau-xi_p)/w) + atan((tau+xi_p)/w)] (odd, analytic),
        denser by about 1/(1 - A/w) around xi = +-xi_p so the rows keep up with a peak-concentrated basis."""
        J = int(round(L / h))
        self.h = h
        if cluster is None:
            xi = lambda t: t; dxi = lambda t: torch.ones_like(t)
        else:
            xp, cw, A = cluster
            tp = xp
            for _ in range(200):                        # tau_p with xi(tau_p) = xp (the symmetrising term shifts it)
                tp = xp + A * math.atan(2 * tp / cw)
            xp = tp
            xi = lambda t: t - A * (torch.atan((t - xp) / cw) + torch.atan((t + xp) / cw))
            dxi = lambda t: 1 - (A / cw) * (1 / (1 + ((t - xp) / cw) ** 2) + 1 / (1 + ((t + xp) / cw) ** 2))
        tau = h * torch.arange(J + 1, dtype=DT, device=device)
        self.ys = torch.sinh(xi(tau))                   # sources
        w = h * dxi(tau) * torch.cosh(xi(tau)); w[0] *= 0.5; w[-1] *= 0.5
        self.w = w
        nc = int(round(Lc / h))
        self.y = torch.sinh(xi(h * (torch.arange(nc, dtype=DT, device=device) + 0.5)))   # collocation
        den = self.y[:, None] ** 2 - self.ys[None, :] ** 2
        self.K = self.w[None, :] / den                  # [nc, J+1]
        self.Y = self.ys[-1]
        g, gw = np.polynomial.legendre.leggauss(n_gl)
        self.g = torch.tensor((g + 1) / 2, dtype=DT, device=device)
        self.gw = torch.tensor(gw / 2, dtype=DT, device=device)


def _MQ(c, s, alpha, beta):
    n = c.shape[0]
    T, d1, d2 = cheb3(2 * s - 1, n)
    M, M1, M2 = T @ c, 2 * (d1 @ c), 4 * (d2 @ c)
    Q = 2 * beta * M - alpha * s * M1
    Q1 = (2 * beta - alpha) * M1 - alpha * s * M2
    return M, Q, Q1


def fields_local(params, y):
    """Theta, Omega, Omega' at points y."""
    lam, c = params[0], params[1:]
    alpha = 1.0 / (1.0 + lam); beta = (1 - alpha) / 2
    r = 1 + y * y
    s = r ** (-alpha / 2)
    M, Q, Q1 = _MQ(c, s, alpha, beta)
    Th = r ** beta * M
    Om = y * r ** (beta - 1) * Q
    Omy = r ** (beta - 1) * Q + y * y * r ** (beta - 2) * (2 * (beta - 1) * Q - alpha * s * Q1)
    return Th, Om, Omy


def velocity(params, G):
    """U, U' at the collocation points, and U'(0)."""
    lam, c = params[0], params[1:]
    alpha = 1.0 / (1.0 + lam); beta = (1 - alpha) / 2
    Th, Om, _ = fields_local(params, G.ys)
    sY = (1 + G.Y * G.Y) ** (-alpha / 2)
    Mt, Qt, _ = _MQ(c, sY * G.g, alpha, beta)
    tM = sY * (G.gw @ Mt) / alpha                       # int_Y^inf y'^(-1-alpha) M dy'
    tQ = sY * (G.gw @ Qt) / alpha
    U = (2 * G.y / math.pi) * (G.K @ Th - tM)
    Up = (2 / math.pi) * (G.K @ (Om * G.ys) - tQ)
    one = torch.ones(1, dtype=Om.dtype, device=Om.device)
    q1 = _MQ(c, one, alpha, beta)[1][0]                 # Omega/y at y = 0
    ratio = torch.cat([q1.reshape(1), Om[1:] / G.ys[1:]])
    Up0 = -(2 / math.pi) * (G.w @ ratio + tQ)
    return U, Up, Up0


def residual_vec(params, G, a=-1.0, lam_row=True):
    lam, c = params[0], params[1:]
    alpha = 1.0 / (1.0 + lam); beta = (1 - alpha) / 2
    _, Om, Omy = fields_local(params, G.y)
    U, Up, Up0 = velocity(params, G)
    R = (Om + ((1 + lam) * G.y + a * U) * Omy - Om * Up) / (G.y * (1 + G.y * G.y) ** (beta - 1))
    one = torch.ones(1, dtype=Om.dtype, device=Om.device)
    rows = [R, (_MQ(c, one, alpha, beta)[1] + 2.0)]
    if lam_row:
        rows.append((lam + 2 - (1 - a) * Up0).reshape(1))
    return torch.cat(rows)


def residual_vec_k(params, G, k=2, a=-1.0):
    """Family with Theta ~ y^(2k) at the origin (hypothesis for the k-1 th unstable profile; k=1 is the
    stable family with a different normalisation). Rows: collocation; Theta''(0)=0 i.e. Q(1)=0 (k=2);
    tail normalisation M(0) = -1; origin exponent row lam = 2k (1 + lam + a U'(0)) ... derived for a=-1:
    -lam Theta + ((1+lam) y - U) Theta' = 0 with Theta ~ y^(2k) gives lam = 2k (1 + lam - U'(0))."""
    assert a == -1.0 and k in (1, 2)
    lam, c = params[0], params[1:]
    alpha = 1.0 / (1.0 + lam); beta = (1 - alpha) / 2
    _, Om, Omy = fields_local(params, G.y)
    U, Up, Up0 = velocity(params, G)
    y2 = G.y * G.y
    R = (Om + ((1 + lam) * G.y - U) * Omy - Om * Up) / (G.y * (1 + y2) ** (beta - 1) * (y2 / (1 + y2)) ** (k - 1))
    one = torch.ones(1, dtype=Om.dtype, device=Om.device)
    sgn = torch.tensor([(-1.0) ** j for j in range(c.shape[0])], dtype=c.dtype, device=c.device)
    rows = [R, ((c * sgn).sum() + 1.0).reshape(1), (lam - 2 * k * (1 + lam - Up0)).reshape(1)]
    if k == 2:
        rows.append(_MQ(c, one, alpha, beta)[1])
    return torch.cat(rows)


# ---- family Omega ~ y^3 at the origin (Theta''(0) = 0 built in exactly) -----------------------
def lift_matrix(n, device="cuda"):
    """A [(n+2) x n]: Chebyshev coefficients of (1-s)^2 * sum_k c_k T_k(2s-1).
    (1-s)^2 = 3/8 - T1/2 + T2/8 in x = 2s-1; T_j T_k = (T_(j+k) + T_|j-k|)/2."""
    A = torch.zeros(n + 2, n, dtype=DT, device=device)
    for k in range(n):
        A[k, k] += 3 / 8
        for j, w in ((1, -0.5), (2, 0.125)):
            A[k + j, k] += w / 2
            A[abs(k - j), k] += w / 2
    return A


def params_k2(q, A):
    """q = [lam, C0, c_N...] -> [lam, c_M...] with M = C0 (1 + (2 beta/alpha)(s-1)) + (1-s)^2 N(s),
    which makes Q(1) = 0, i.e. Omega'(0) = 0, identically."""
    lam, C0, cN = q[0], q[1], q[2:]
    alpha = 1.0 / (1.0 + lam); beta = (1 - alpha) / 2
    cM = A @ cN
    e0 = torch.zeros_like(cM); e0[0] = 1.0
    e1 = torch.zeros_like(cM); e1[1] = 1.0
    cM = cM + C0 * ((1 - beta / alpha) * e0 + (beta / alpha) * e1)
    return torch.cat([lam.reshape(1), cM])


def residual_vec_k2(q, G, A):
    """Rows: collocation (normalised by the y^3 behaviour), tail normalisation M(0) = -1,
    origin exponent row lam = 4 (1 + lam - U'(0))."""
    p = params_k2(q, A)
    lam, c = p[0], p[1:]
    alpha = 1.0 / (1.0 + lam); beta = (1 - alpha) / 2
    _, Om, Omy = fields_local(p, G.y)
    U, Up, Up0 = velocity(p, G)
    y2 = G.y * G.y
    R = (Om + ((1 + lam) * G.y - U) * Omy - Om * Up) / (G.y * (1 + y2) ** (beta - 1) * (y2 / (1 + y2)))
    sgn = torch.tensor([(-1.0) ** j for j in range(c.shape[0])], dtype=c.dtype, device=c.device)
    return torch.cat([R, ((c * sgn).sum() + 1.0).reshape(1), (lam - 4 * (1 + lam - Up0)).reshape(1)])
