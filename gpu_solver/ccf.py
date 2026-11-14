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
    def __init__(self, h=0.1, L=40.0, Lc=22.0, n_gl=64, device="cuda"):
        J = int(round(L / h))
        self.h = h
        xi = h * torch.arange(J + 1, dtype=DT, device=device)
        self.ys = torch.sinh(xi)                        # sources
        w = h * torch.cosh(xi); w[0] *= 0.5; w[-1] *= 0.5
        self.w = w
        nc = int(round(Lc / h))
        self.y = torch.sinh(h * (torch.arange(nc, dtype=DT, device=device) + 0.5))   # collocation
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
