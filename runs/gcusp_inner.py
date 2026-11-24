"""Inner problem of the cusp at an interior stagnation point for the generalised CLM, general a < 0
(generalises runs/cusp_inner.py, which is the case a = -1).

Derivation (formal). Near y* with V(y*) = V* -> 0+, V = (1+lam) y + a U: inside the core the singular terms dominate,
    V' = a U' (+ (1+lam), relative size 1/peak),   V Omega' = Omega U' (+ Omega, relative size 1/peak),
so V Omega' = Omega V'/a, i.e. Omega = -P (V/V*)^(1/a), P = peak of |Omega|.  With y - y* = L xi, V = V* v(xi),
L = V*/(|a| P), U' = H Omega:
    v'(xi) = H(v^(-q)),   q = -1/a = 1/|a|,   v(0) = 1,  v even,  v ~ c |xi|^(1-p)  (p = 1/(1-a)),
    c^(1+q) = tan(pi p/2)/(1-p)        [H(|x|^-p) = tan(pi p/2) sgn(x) |x|^-p]
a = -1: q = 1, p = 1/2, c = sqrt 2: v' = H(1/v), v ~ sqrt(2|xi|)  (cusp_inner.py).
Far field: v = c xi^(1-p) (1 + e xi^(-s)) gives  (1-p-s) tan(pi p/2) + p tan(pi (p+s)/2) = 0; with s = 1-p+t:
    t tan(pi p/2) + p cot(pi t/2) = 0,  t = +-i w:   w tanh(pi w/2) = p cot(pi p/2)       (a = -1: 2 w tanh(pi w/2) = 1)
so v - c xi^(1-p) = C cos(w ln xi + phase) + ... (O(1) absolute, as at a = -1).
Method as cusp_inner.py: g = v^(-q) = g0 + g1, explicit pair g0 = A rho^-p cos(p phi)/cos(pi p/2),
H g0 = A rho^-p sin(p phi)/cos(pi p/2) (phi = atan(xi/b), boundary values of A e^(i pi p/2) (z + i b)^-p), A = c^-q;
g1 on xi = exp(tau_j), H g1 by the trapezoid rule in tau, v by the midpoint rule, residual g v^q - 1, Newton.
Half-width: |Omega| = P/2 where v = 2^(1/q) = 2^|a|; prints xi_h there and the far-field frequency fitted freely.
Usage: gcusp_inner.py a [h]     (CPU)"""
import sys, math, torch
import numpy as np
torch.set_default_dtype(torch.float64)
a = float(sys.argv[1]) if len(sys.argv) > 1 else -1.0
h = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
q = -1.0 / a; p = 1.0 / (1.0 - a); b = 1.0
c = (math.tan(math.pi * p / 2) / (1 - p)) ** (1 / (1 + q)); A = c ** (-q); cp = math.cos(math.pi * p / 2)


def wroot():
    f = lambda w: w * math.tanh(math.pi * w / 2) - p / math.tan(math.pi * p / 2)
    lo, hi = 1e-9, 20.0
    for _ in range(200):
        m = (lo + hi) / 2; lo, hi = (m, hi) if f(m) < 0 else (lo, m)
    return (lo + hi) / 2


W = wroot()
tau = torch.arange(math.log(1e-7), math.log(1e9), h); ys = torch.exp(tau); yc = torch.exp(tau[:-1] + h / 2); n = ys.numel()
rho = lambda x: torch.sqrt(x * x + b * b)
phi = lambda x: torch.atan(x / b)
g0 = lambda x: A * rho(x) ** (-p) * torch.cos(p * phi(x)) / cp
Hg0 = lambda x: A * rho(x) ** (-p) * torch.sin(p * phi(x)) / cp
K = (2 * yc[:, None] / math.pi) * (h * ys)[None, :] / (yc[:, None] ** 2 - ys[None, :] ** 2)


def vof(g1):
    dv = (Hg0(yc) + K @ g1) * h * yc
    return 1 + torch.cat([torch.zeros(1), torch.cumsum(dv, 0)])


def res(g1):
    v = vof(g1)
    return (g0(ys) + g1) * v.clamp_min(1e-300) ** q - 1


g1 = (1 + c ** (1 / (1 - p)) * ys) ** (-q * (1 - p)) - g0(ys)
for it in range(60):
    r = res(g1); J = torch.autograd.functional.jacobian(res, g1); st = torch.linalg.solve(J, -r); t = 1.0
    while t > 1e-4:
        gn = g1 + t * st
        if (vof(gn) > 0).all() and res(gn).norm() < r.norm():
            break
        t /= 2
    g1 = gn; rn = res(g1).abs().max().item()
    if rn < 1e-13:
        break
v = vof(g1); g = g0(ys) + g1


def cross(level):
    i = (v > level).nonzero()[0].item(); f = (level - v[i - 1]) / (v[i] - v[i - 1]); return math.exp(tau[i - 1].item() + f.item() * h)


lev = 2.0 ** abs(a); xh = cross(lev)
print(f"a {a} p {p:.6f} q {q:.6f} c {c:.6f}  w(theory) {W:.6f}   h {h} n {n} its {it + 1} res {rn:.1e}")
print(f"  xi_h (v = 2^|a| = {lev:.4f}, |Omega| = peak/2) {xh:.5f};  xi at v = 2: {cross(2.0):.5f}")
for x in (1e2, 1e4, 1e6):
    i = (ys - x).abs().argmin().item(); print(f"   xi {ys[i].item():9.3g}  v/(c xi^(1-p)) {v[i].item() / (c * ys[i].item() ** (1 - p)):.5f}")
# far field: u = v - c xi^(1-p) against cos/sin(w ln xi), w scanned freely
for lo, hi in ((1e2, 1e7), (1e3, 1e7)):
    m = (ys > lo) & (ys < hi); x = ys[m].numpy(); u = (v[m] - c * ys[m] ** (1 - p)).numpy(); lx = np.log(x)
    best = None
    for ww in np.linspace(0.2, 1.5, 2601):
        M = np.stack([np.cos(ww * lx), np.sin(ww * lx), np.ones_like(x)], 1); cc, *_ = np.linalg.lstsq(M, u, rcond=None)
        rr = np.sqrt(np.mean((M @ cc - u) ** 2))
        if best is None or rr < best[0]:
            best = (rr, ww, cc)
    M = np.stack([np.cos(W * lx), np.sin(W * lx), np.ones_like(x)], 1); cc, *_ = np.linalg.lstsq(M, u, rcond=None)
    rW = np.sqrt(np.mean((M @ cc - u) ** 2))
    print(f"  far field {lo:.0e}..{hi:.0e}: free fit w {best[1]:.4f} rms {best[0]:.1e} amp {math.hypot(*best[2][:2]):.4f} const {best[2][2]:+.3f};"
          f" at w(theory): rms {rW:.1e} (rms of u {np.sqrt(np.mean(u * u)):.3f})")
