"""Test of a heuristic small-|a| description of the first unstable branch (see STATE.md 2c).
Outer solution: pole pair G = U' - i Omega = u0/(1 - z^2) (X = y/y*), poles at distance eps from the real axis. Three relations:
  (origin, exact)    u0 = (1 + gamma)/(1 - a),                       gamma = 1 + lam
  (double pole)      gamma + a (u0/2) log(2/eps) = u0/4
  (simple pole)      gamma + a u0/4 = 1 - u0/4
=> gamma_pred(a) = (1 - rho)/(1 + rho), rho = (1 + a)/(4 (1 - a));  gamma -> 3/5 (lam_1 -> -2/5), u0 -> 8/5,
   |a| log(2/eps) -> 1/4 as a -> 0-."""
import torch, math, sys, numpy as np
from gpu_solver import ccf as C, ccf_fast as F, gclm_log as L
torch.set_num_threads(6); rows = []
for f in sys.argv[1:]:
    for x in torch.load(f):
        if "ypk" not in x or x["a"] < -0.42: continue
        p = x["p"]; F.MAP = tuple(x["map"]); n = p.numel() - 2; yp = x["ypk"]
        y = yp * torch.exp(torch.linspace(-0.6, 0.6, 200001, dtype=C.DT)); Om = (L._basis(p[0], y, n)[1] @ p[1:]).abs()
        i = Om.argmax(); pk = Om[i].item(); hw = (Om > pk / 2).nonzero().flatten()
        rows.append((x["a"], p[0].item(), pk, y[i].item(), (y[hw[-1]] - y[hw[0]]).item()))
R = np.array(sorted(set(rows))); a, lam, pk, ys, fw = R.T; g = 1 + lam; u0 = (1 + g) / (1 - a); eps = fw / (2 * ys)
rho = (1 + a) / (4 * (1 - a)); gp = (1 - rho) / (1 + rho)
kap_meas = -a * np.log(2 / eps); kap_pred = 2 * (g - u0 / 4) / u0
print("   a        gamma    gamma_pred  diff      diff/a^2 | eps=HWHM/y*  |a|log(2/eps)  2(g-u0/4)/u0  | peak*eps (pred u0/2)   u0")
for k in list(range(0, len(a), max(1, len(a) // 22))) + [len(a) - 1]:
    print(f"{a[k]:+.5f}  {g[k]:.5f}  {gp[k]:.5f}   {g[k]-gp[k]:+.5f}  {(g[k]-gp[k])/a[k]**2:6.2f}  | {eps[k]:.3e}   {kap_meas[k]:.4f}        {kap_pred[k]:.4f}       | {pk[k]*fw[k]/2/ys[k]*1.0:.4f} ({u0[k]/2:.4f})   {u0[k]:.4f}")
k = a > -0.12; d = g[k] - gp[k]
for form, M in (("c1 a^2", np.c_[a[k] ** 2]), ("c1 a + c2 a^2", np.c_[a[k], a[k] ** 2]), ("c0 + c1 a + c2 a^2", np.c_[np.ones(k.sum()), a[k], a[k] ** 2])):
    c = np.linalg.lstsq(M, d, rcond=None)[0]; print(f"gamma - gamma_pred on a > -0.12 fitted by {form}: coeffs {np.round(c, 4)} rms {np.sqrt(np.mean((M @ c - d) ** 2)):.1e}")
c = np.polyfit(a[k], g[k], 3); print("cubic extrapolation of gamma to a = 0:", round(float(np.polyval(c, 0.0)), 4), "(prediction 0.6)")
