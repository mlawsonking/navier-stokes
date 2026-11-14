"""Exploration tool (second-order accurate, not the precision solver).
CCF in Burgers form: -lam Theta + ((1+lam) y - U) Theta' = 0, U = H Theta, Theta even.
For given lam iterate  Theta -> U = H Theta -> Theta = -y^mu exp(int_0^y [lam/((1+lam)y'-U) - mu/y'] dy'),
mu = lam/(1+lam-U'(0)), and report mu(lam). A profile is smooth at 0 iff mu is an even integer.
Usage: ccf_picard.py h L lam1,lam2,...   (numpy, CPU)"""
import sys, json, numpy as np
h, L = float(sys.argv[1]), float(sys.argv[2])
lams = [float(x) for x in sys.argv[3].split(",")]
J = int(round(L / h))
xi = h * np.arange(J + 1); ys = np.sinh(xi); w = h * np.cosh(xi); w[0] *= 0.5; w[-1] *= 0.5
xm = h * (np.arange(J) + 0.5); ym = np.sinh(xm)
K = w[None, :] / (ym[:, None] ** 2 - ys[None, :] ** 2)
Y = ys[-1]


def step(Th, lam):
    p = lam / (1 + lam)
    A = Th[-1] / Y ** p
    tail = -A * Y ** (p - 1) / (1 - p)
    U = (2 * ym / np.pi) * (K @ Th + tail)
    f = np.zeros_like(ys); f[1:] = Th[1:] / ys[1:] ** 2
    f[0] = 2 * f[1] - f[2]                      # crude extrapolation (exact only for mu = 2)
    u0 = -(2 / np.pi) * (w @ f + tail)
    mu = lam / (1 + lam - u0)
    g = lam / ((1 + lam) * ym - U) * np.cosh(xm) - mu * np.cosh(xm) / np.sinh(xm)
    lnE = np.concatenate([[0.0], np.cumsum(h * g)])
    Thn = -ys ** mu * np.exp(lnE)
    return Thn, mu, u0, ((1 + lam) * ym - U).min()


out = open("runs/ccf_picard.jsonl", "a")
Th = -ys ** 2 / (1 + ys ** 2) ** 0.75
for lam in lams:
    for it in range(400):
        Thn, mu, u0, vmin = step(Th, lam)
        if not np.all(np.isfinite(Thn)) or vmin <= 0:
            print(f"lam {lam:.4f} FAILED at it {it} (min transport speed {vmin:.3g})"); Th = -ys ** 2 / (1 + ys ** 2) ** 0.75; break
        d = np.abs(np.log(Thn[1:] / Th[1:])).max()
        Th = -np.exp(0.5 * np.log(-Th[1:]) + 0.5 * np.log(-Thn[1:])); Th = np.concatenate([[0.0], Th])
        if d < 1e-11:
            break
    else:
        it = 400
    print(f"lam {lam:.5f} mu {mu:.6f} U'(0) {u0:.6f} its {it} change {d:.1e} tail A {Th[-1]/Y**(lam/(1+lam)):.5f}", flush=True)
    out.write(json.dumps(dict(case="2a-ccf-picard", h=h, L=L, lam=lam, mu=mu, Up0=u0, iters=it, last_change=d)) + "\n")
