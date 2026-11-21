"""Does the non-smooth CCF family approach its singular (cusp) end in a SPIRAL?  (STATE 2e)
Linearising the inner equation v' = H(1/v) about v0 = sqrt(2|xi|) with v = v0 (1 + xi^(-s)) gives
2s - 1 = tan(pi/4 + pi s/2): s = 1/2 +- 0.649424 i (and real 2.330, -1.330). In outer variables the correction
is  V* cos(2 w ln V* + phase), w = 0.649424, so every global quantity q should behave like
    q = q_c + sigma [a cos(2 w ln sigma) + b sin(2 w ln sigma)] (+ c sigma^2),   sigma = 1/peak.
Peaks are re-measured on a fine local grid (the tracker's fixed grid was too coarse above peak 40).
Usage: family_spiral_fit.py file.pt [file.pt ...]"""
import sys, math, json, torch, numpy as np
from gpu_solver import ccf_sing as S, ccf_fast as F
W = 0.649424; rows = {}
for f in sys.argv[1:]:
    for x in torch.load(f):
        p = x["p"].double(); nj, J = x["njJ"]; sh = (p.numel() - 2 - nj * J, nj, J); F.MAP = tuple(x["map"]); y0 = x["ypk"]; w = 0.02
        for _ in range(8):
            y = torch.linspace(y0 - w, y0 + w, 2001, dtype=S.DT); BTh, BOm, *_ = S._fields(p[0], p[1], y, sh); Om = (BOm @ p[2:]).abs(); i = Om.argmax()
            y0 = y[i].item(); pk = Om[i].item(); hw = (Om > pk / 2).nonzero().flatten(); a, b = y[hw[0]].item(), y[hw[-1]].item(); th = abs((BTh @ p[2:])[i].item())
            if hw[0] > 0 and hw[-1] < 2000 and hw.numel() > 150: break
            w = 3 * (b - a) if (hw[0] > 0 and hw[-1] < 2000) else 4 * w
        lam, mu = p[0].item(), p[1].item()
        rows[round(lam, 9)] = dict(lam=lam, mu=mu, up0=1 + lam - lam / mu, peak=pk, ystar=y0, hw=(b - a) / 2, kappa=lam * th)
R = sorted(rows.values(), key=lambda r: -r["lam"])
print("  lam        mu         peak     y*       kappa=lam|Th*|  peak^2 HW   / kappa (inner problem: 3.35819)")
for r in R[:: max(1, len(R) // 12)] + [R[-1]]:
    print(f"  {r['lam']:.6f} {r['mu']:.6f} {r['peak']:8.3f} {r['ystar']:.5f} {r['kappa']:.5f}  {r['peak']**2*r['hw']:.4f}  {r['peak']**2*r['hw']/r['kappa']:.5f}")
json.dump(R, open("runs/family_end_table.json", "w"))
def fit(key, pmin, model):
    d = [r for r in R if r["peak"] > pmin]; s = np.array([1 / r["peak"] for r in d]); q = np.array([r[key] for r in d]); L = 2 * W * np.log(s)
    cols = {"poly2": [np.ones_like(s), s, s * s], "poly3": [np.ones_like(s), s, s * s, s ** 3], "spiral": [np.ones_like(s), s * np.cos(L), s * np.sin(L)],
            "spiral+s2": [np.ones_like(s), s * np.cos(L), s * np.sin(L), s * s]}[model]
    M = np.stack(cols, 1); c, *_ = np.linalg.lstsq(M, q, rcond=None); rms = np.sqrt(np.mean((M @ c - q) ** 2))
    return c, rms, len(d)
for key in ("lam", "mu", "up0", "ystar"):
    print(key)
    for pmin in (12, 20, 30):
        out = []
        for model in ("poly2", "spiral", "poly3", "spiral+s2"):
            c, rms, n = fit(key, pmin, model); out.append(f"{model}: q_c {c[0]:.6f} rms {rms:.1e}")
        print(f"  peak > {pmin} (n={n})  " + " | ".join(out))
# where does the spiral model put the turning points of lam and the extrema / crossings of mu?
for pmin in (12, 20, 30):
    for model in ("spiral", "spiral+s2"):
        cl, _, _ = fit("lam", pmin, model); cm, _, _ = fit("mu", pmin, model)
        s = np.exp(np.linspace(math.log(1 / 60), math.log(1e-5), 400001)); L = 2 * W * np.log(s)
        ev = lambda c: c[0] + s * (c[1] * np.cos(L) + c[2] * np.sin(L)) + (c[3] * s * s if len(c) > 3 else 0)
        lam, mu = ev(cl), ev(cm); dl = np.diff(lam); folds = np.nonzero(dl[:-1] * dl[1:] < 0)[0][:3]; cr = np.nonzero((mu[:-1] - 2) * (mu[1:] - 2) < 0)[0][:3]
        print(f"peak > {pmin} {model}: lam folds at peak " + ", ".join(f"{1/s[i]:.0f} (lam {lam[i]:.6f}, mu {mu[i]:.5f})" for i in folds) + f" | min mu for peak > 60: {mu.min():.5f} at peak {1/s[mu.argmin()]:.0f} | mu = 2 at peak: " + (", ".join(f"{1/s[i]:.0f} (lam {lam[i]:.6f})" for i in cr) or "never"))
