"""Universal inner problem of the cusp at a stagnation point (CCF, a = -1; see STATE 2d).
Inner variables: y - y* = (V*^2/kappa) xi, V = V* v(xi), -Omega = (kappa/V*) / v, kappa = lam |Theta*|. Leading order:
    v'(xi) = H(1/v),   v(0) = 1,  v even,  v ~ sqrt(2 |xi|) at infinity       (no parameter left)
Prediction to test against the family: peak^2 x half-width = kappa * xi_h, where v(xi_h) = 2.
Method: w = 1/v = w0 + w1 with the explicit pair w0 = A sqrt(rho + b)/rho, H w0 = A sgn(xi) sqrt(rho - b)/rho
(rho = sqrt(xi^2 + b^2), A = 1/sqrt 2; boundary values of C (z + i b)^(-1/2)), w1 unknown on xi = exp(tau_j);
H w1 at the half steps by the trapezoid rule in tau (even function), v by the midpoint rule; Newton.
Usage: cusp_inner.py [h]"""
import sys, math, torch
torch.set_default_dtype(torch.float64)
h = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02; b = 1.0; A = 1 / math.sqrt(2)
tau = torch.arange(math.log(1e-7), math.log(1e9), h); ys = torch.exp(tau); yc = torch.exp(tau[:-1] + h / 2); n = ys.numel()
rho = lambda x: torch.sqrt(x * x + b * b)
w0 = lambda x: A * torch.sqrt(rho(x) + b) / rho(x)
Hw0 = lambda x: A * x / torch.sqrt(rho(x) + b) / rho(x)          # sqrt(rho - b) = x / sqrt(rho + b), stable at small x
K = (2 * yc[:, None] / math.pi) * (h * ys)[None, :] / (yc[:, None] ** 2 - ys[None, :] ** 2)
def vof(w1):
    dv = (Hw0(yc) + K @ w1) * h * yc
    return 1 + torch.cat([torch.zeros(1), torch.cumsum(dv, 0)])
res = lambda w1: (w0(ys) + w1) * vof(w1) - 1
w1 = 1 / torch.sqrt(1 + 2 * ys) - w0(ys)
for it in range(40):
    r = res(w1); J = torch.autograd.functional.jacobian(res, w1); st = torch.linalg.solve(J, -r); t = 1.0
    while t > 1e-4 and not res(w1 + t * st).norm() < r.norm(): t /= 2
    w1 = w1 + t * st; rn = res(w1).abs().max().item()
    if rn < 1e-13: break
v = vof(w1); w = w0(ys) + w1
def cross(level):
    i = (v > level).nonzero()[0].item(); f = (level - v[i - 1]) / (v[i] - v[i - 1]); return math.exp(tau[i - 1].item() + f.item() * h)
xh = cross(2.0)
print(f"h {h} n {n} newton its {it + 1} res {rn:.1e}  xi_h (v = 2) {xh:.5f}   naive cut-off value 0.5")
print("  v(xi)/sqrt(2 xi) and  (1/v) sqrt(2 xi)  [= Omega sqrt(d)/A]:")
for x in (1, 3, 10, 30, 100, 1e3, 1e4, 1e6):
    i = (ys - x).abs().argmin().item(); print(f"   xi {ys[i].item():10.3g}  v {v[i].item():10.4f}  v/sqrt(2xi) {v[i].item()/math.sqrt(2*ys[i].item()):.4f}  w sqrt(2 xi) {w[i].item()*math.sqrt(2*ys[i].item()):.4f}  w1/w0 {w1[i].item()/w0(ys[i:i+1]).item():+.2e}")
i = (w * torch.sqrt(ys)).argmax().item(); print(f"  max of w sqrt(2 xi): {w[i].item()*math.sqrt(2*ys[i].item()):.4f} at xi = {ys[i].item():.3f} = {ys[i].item()/xh:.2f} half-widths;  w1 at ends {w1[0].item():+.3e} {w1[-1].item():+.3e}")
# far field: v - sqrt(2 xi) should be C cos(w ln xi + phase), w = 0.649424 (2 w tanh(pi w/2) = 1), from the linearisation
import numpy as np
W = 0.649424
for lo, hi in ((1e2, 1e7), (1e3, 1e7), (1e2, 1e5)):
    m = (ys > lo) & (ys < hi); x = ys[m].numpy(); u = (v[m] - torch.sqrt(2 * ys[m])).numpy(); L = W * np.log(x)
    for name, cols in (("osc", [np.cos(L), np.sin(L)]), ("osc+const", [np.cos(L), np.sin(L), np.ones_like(x)]), ("const+xi^-1/2", [np.ones_like(x), x ** -0.5])):
        M = np.stack(cols, 1); c, *_ = np.linalg.lstsq(M, u, rcond=None); print(f"  far field {lo:.0e}..{hi:.0e} {name}: coeffs {np.round(c, 4)} rms {np.sqrt(np.mean((M @ c - u) ** 2)):.1e} (rms of u {np.sqrt(np.mean(u * u)):.2f})")
