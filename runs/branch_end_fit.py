"""Where does the first unstable branch end? Fit FWHM/y*, 1/peak, q0, y* against (a_c - a)^p on the tail."""
import torch, math, sys, numpy as np
from gpu_solver import ccf as C, ccf_fast as F, gclm_log as L
torch.set_num_threads(6); rows = []
for f in sys.argv[1:]:
    for x in torch.load(f):
        if "ypk" not in x or x["a"] < -0.2: continue
        p = x["p"]; F.MAP = tuple(x["map"]); n = p.numel() - 2; yp = x["ypk"]
        y = yp * torch.exp(torch.linspace(-0.6, 0.6, 200001, dtype=C.DT)); Om = (L._basis(p[0], y, n)[1] @ p[1:]).abs()
        i = Om.argmax(); pk = Om[i].item(); hw = (Om > pk / 2).nonzero().flatten()
        rows.append((x["a"], p[0].item(), p[1].item(), pk, y[i].item(), (y[hw[-1]] - y[hw[0]]).item()))
R = np.array(sorted(set(rows))); a, lam, q0, pk, ys, fw = R.T
print("points", len(a), "a range", a[0], a[-1])
def fit(v, name, lo=-0.13):
    k = a > lo; best = None
    for ac in np.linspace(-0.065, 0.02, 171):
        if ac <= a[k].max(): continue
        X = np.log(ac - a[k]); Y = np.log(np.abs(v[k])); c = np.polyfit(X, Y, 1); res = np.sqrt(np.mean((np.polyval(c, X) - Y) ** 2))
        if best is None or res < best[0]: best = (res, ac, c[0])
    print(f"{name:10s} tail a > {lo}: best a_c {best[1]:+.4f} exponent {best[2]:+.3f} rms log-residual {best[0]:.1e}")
    for ac in (-0.04, -0.02, 0.0):
        X = np.log(ac - a[k]); Y = np.log(np.abs(v[k])); c = np.polyfit(X, Y, 1); print(f"     forced a_c {ac:+.2f}: exponent {c[0]:+.3f} rms {np.sqrt(np.mean((np.polyval(c, X) - Y) ** 2)):.1e}")
for lo in (-0.13, -0.10):
    fit(fw / ys, "FWHM/y*", lo); fit(1 / pk, "1/peak", lo); fit(q0, "q0", lo); fit(ys, "y*", lo)
k = a > -0.10
print("peak*FWHM/y* on the tail:", np.round((pk * fw / ys)[k][::6], 4))
print("lam + a on the tail:", np.round((lam + a)[k][::6], 5), " U'(0):", np.round(((lam + 2) / (1 - a))[k][::6], 5))
c = np.polyfit(a[k], lam[k], 2); print("lam extrapolated (quadratic in a) to a = -0.04, -0.02, 0:", [round(float(np.polyval(c, v)), 4) for v in (-0.04, -0.02, 0.0)])
