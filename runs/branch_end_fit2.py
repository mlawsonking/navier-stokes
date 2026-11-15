"""Second look at the end of the first unstable branch: finite a_c power law  q ~ (a_c - a)^e   versus an essential
singularity at a = 0,  q ~ |a|^p exp(-C/|a|)  (a pole whose distance to the real axis is exponentially small in 1/|a|,
as a log-divergent a U at the pole would give). Same data as branch_end_fit.py."""
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
R = np.array(sorted(set(rows))); a, lam, q0, pk, ys, fw = R.T; print("points", len(a), "a range", a[0], a[-1])
def rms(M, Y): c = np.linalg.lstsq(M, Y, rcond=None)[0]; return np.sqrt(np.mean((M @ c - Y) ** 2)), c
def cmp(v, name, lo, hi=0.0):
    k = (a > lo) & (a < hi); Y = np.log(np.abs(v[k])); A = a[k]; one = np.ones_like(A)
    best = min(((rms(np.c_[one, np.log(ac - A)], Y)[0], ac) for ac in np.linspace(-0.07, 0.03, 401) if ac > A.max()))
    r2, c2 = rms(np.c_[one, 1 / A], Y); r3, c3 = rms(np.c_[one, 1 / A, np.log(-A)], Y)
    print(f"{name:9s} window ({lo:+.3f},{hi:+.3f}) n={k.sum():3d}: power law a_c {best[1]:+.4f} rms {best[0]:.1e} | exp(-C/|a|): C {c2[1]:.4f} rms {r2:.1e} | |a|^p exp(-C/|a|): C {c3[1]:.4f} p {c3[2]:+.2f} rms {r3:.1e}")
    return c3
for name, v in (("FWHM/y*", fw / ys), ("1/peak", 1 / pk), ("q0", q0), ("y*", ys)):
    for lo, hi in ((-0.13, 0.0), (-0.10, 0.0), (-0.085, 0.0), (-0.13, -0.085)):
        cmp(v, name, lo, hi)
# out-of-sample: fit on a < -0.085, predict the last point
k = (a > -0.13) & (a < -0.085); A = a[k]; one = np.ones_like(A)
for name, v in (("FWHM/y*", fw / ys), ("1/peak", 1 / pk)):
    Y = np.log(np.abs(v[k])); tgt = math.log(abs(v[-1])); al = a[-1]
    best = min(((rms(np.c_[one, np.log(ac - A)], Y), ac) for ac in np.linspace(-0.08, 0.03, 441) if ac > A.max()), key=lambda t: t[0][0])
    (rp, cp), ac = best; pp = cp[0] + cp[1] * math.log(ac - al) if ac > al else float("nan")
    r3, c3 = rms(np.c_[one, 1 / A, np.log(-A)], Y); pe = c3[0] + c3[1] / al + c3[2] * math.log(-al)
    r2, c2 = rms(np.c_[one, 1 / A], Y); pe2 = c2[0] + c2[1] / al
    print(f"{name}: fit on -0.13<a<-0.085, predict log at a={al:.4f}: actual {tgt:.3f}; power law (a_c {ac:+.4f}) {pp:.3f}; exp 2-par {pe2:.3f}; exp 3-par {pe:.3f}")
print("lam on tail:", np.round(lam[a > -0.10][::8], 5)); print("a   on tail:", np.round(a[a > -0.10][::8], 5))
