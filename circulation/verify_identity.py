#!/usr/bin/env python3
"""Numerical check of the circulation / heat-kernel identity.

Claim.  For a smooth, decaying, divergence-free field u on R^3 with vorticity
w = curl u, any point x, unit vector n and time t > 0,

    4 pi t * (e^{t Lap} w)(x) . n  =  E[ Gamma(C) ],

where C is a random circle with axis through x in direction n: its centre is
x + h n with h ~ Normal(0, 2t), its radius rho has density (rho/2t) exp(-rho^2/4t),
and Gamma(C) is the circulation of u around C (right-handed about n).

The test field is u = curl A with A a sum of two Gaussian bumps, so the heat
flow of w is known in closed form and the left side needs no quadrature.
"""
from __future__ import annotations

import numpy as np
import sympy as sp

x1, x2, x3, t = sp.symbols("x1 x2 x3 t", real=True)
X = sp.Matrix([x1, x2, x3])


def curl(F):
    return sp.Matrix([
        sp.diff(F[2], x2) - sp.diff(F[1], x3),
        sp.diff(F[0], x3) - sp.diff(F[2], x1),
        sp.diff(F[1], x1) - sp.diff(F[0], x2),
    ])


def heated_gaussian(amp, centre, sigma):
    """amp * exp(-|x-c|^2 / 2 sigma^2) after heat flow for time t (exact)."""
    s2 = sigma ** 2 + 2 * t
    r2 = sum((X[i] - centre[i]) ** 2 for i in range(3))
    return sp.Matrix(amp) * (sigma ** 2 / s2) ** sp.Rational(3, 2) * sp.exp(-r2 / (2 * s2))


A_t = heated_gaussian([1.0, -0.6, 0.3], [0.2, -0.1, 0.4], 0.7) + heated_gaussian([-0.4, 0.9, 1.1], [-0.5, 0.3, -0.2], 1.1)
u_t = curl(A_t)
w_t = curl(u_t)
u0 = sp.lambdify((x1, x2, x3), u_t.subs(t, 0), "numpy")
w_heat = sp.lambdify((x1, x2, x3, t), w_t, "numpy")


def frame(n):
    n = np.asarray(n, float) / np.linalg.norm(n)
    a = np.cross(n, [1.0, 0.0, 0.0])
    if np.linalg.norm(a) < 1e-6:
        a = np.cross(n, [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    return n, a, np.cross(n, a)


def circulation(centre, n, rho, m=128):
    """Right-handed circulation of u0 around the circle (trapezoid rule, spectrally accurate)."""
    n, a, b = frame(n)
    th = 2 * np.pi * np.arange(m) / m
    pts = centre[:, None] + rho * (np.outer(a, np.cos(th)) + np.outer(b, np.sin(th)))
    tang = -np.outer(a, np.sin(th)) + np.outer(b, np.cos(th))
    u = np.asarray(u0(pts[0], pts[1], pts[2]), float).reshape(3, m)
    return float(np.sum(u * tang) * rho * 2 * np.pi / m)


def averaged_circulation(x, n, tt, nh=90, nr=90):
    n = np.asarray(n, float) / np.linalg.norm(n)
    hs, hw = np.polynomial.hermite_e.hermegauss(nh)      # weight exp(-z^2/2)
    hw = hw / np.sqrt(2 * np.pi)
    ss, sw = np.polynomial.laguerre.laggauss(nr)          # s = rho^2/4t ~ Exp(1)
    total = 0.0
    for z, wz in zip(hs, hw):
        centre = np.asarray(x, float) + np.sqrt(2 * tt) * z * n
        for s, ws in zip(ss, sw):
            total += wz * ws * circulation(centre, n, np.sqrt(4 * tt * s))
    return total


def main():
    rng = np.random.default_rng(1)
    print(f"{'t':>6} {'4 pi t (e^tLap w).n':>22} {'E[circulation]':>18} {'rel. diff':>10}")
    worst = 0.0
    for _ in range(8):
        x = rng.normal(size=3) * 0.6
        n = rng.normal(size=3); n /= np.linalg.norm(n)
        tt = float(10 ** rng.uniform(-2, 0.5))
        lhs = 4 * np.pi * tt * float(np.dot(np.asarray(w_heat(*x, tt), float).ravel(), n))
        rhs = averaged_circulation(x, n, tt)
        rel = abs(lhs - rhs) / max(abs(lhs), 1e-300)
        worst = max(worst, rel)
        print(f"{tt:6.3f} {lhs:22.12e} {rhs:18.12e} {rel:10.2e}")
    print(f"worst relative difference: {worst:.2e}")

    # Consequence: sup_t 4 pi t |e^{t Lap} w| <= sup over circles of |circulation|.
    lhs_sup = 0.0
    for _ in range(4000):
        x = rng.normal(size=3) * 1.2
        tt = float(10 ** rng.uniform(-3, 1.5))
        lhs_sup = max(lhs_sup, 4 * np.pi * tt * float(np.linalg.norm(np.asarray(w_heat(*x, tt), float))))
    circ_sup = 0.0
    for _ in range(20000):
        c = rng.normal(size=3) * 1.2
        n = rng.normal(size=3)
        circ_sup = max(circ_sup, abs(circulation(c, n, float(10 ** rng.uniform(-1.5, 0.8)))))
    print(f"sampled sup_t 4 pi t |e^(t Lap) w|   = {lhs_sup:.6f}")
    print(f"sampled sup over circles |Gamma|    = {circ_sup:.6f}   (a lower bound for the true sup)")
    assert worst < 1e-8
    assert lhs_sup <= circ_sup


if __name__ == "__main__":
    main()
