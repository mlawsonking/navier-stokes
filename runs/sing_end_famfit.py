"""Family (sing_track4 ledger) against the direct end values (sing_end, A_R = A_L member).
Model: q(P) = q_c + s [a + b cos(2 w ln s) + c sin(2 w ln s)] + d s^2, s = 1/peak, w = 0.649424 (inner value).
The non-oscillating 'a' term is expected from a drift ALONG the singular family; its ratios a_mu/a_lam, a_y/a_lam
should then equal the tangent of that family (dmu/dlam = -0.0430, dy*/dlam = -2.33 from fixed-lam solves).
Usage: sing_end_famfit.py [peak_min]"""
import sys, json
import numpy as np
pmin = float(sys.argv[1]) if len(sys.argv) > 1 else 300
direct = dict(lam=0.4535843163, mu=2.00577172004, y_peak=7.243737072, up0=1.2274447652)
rows = [json.loads(l) for l in open("runs/sing_track.jsonl")]
rows = [r for r in rows if r.get("case", "").startswith("sing-track4-") and r["case"][-2:] in ("w1", "w2", "v1", "v2", "v3", "v4") and r.get("res_offgrid", 1) < 3e-7]
rows.sort(key=lambda r: r["peak"]); out = []
for r in rows:
    if out and abs(r["peak"] / out[-1]["peak"] - 1) < 0.02: continue
    out.append(r)
P = np.array([r["peak"] for r in out]); m = P > pmin; s = 1 / P[m]; ph = 2 * 0.649424 * np.log(s)
print(f"{m.sum()} family states, peak {P[m].min():.0f} .. {P[m].max():.0f}")
A = {}
for k in ("lam", "mu", "y_peak", "up0"):
    q = np.array([r[k] for r in out])[m]
    base = [s, s * np.cos(ph), s * np.sin(ph), s * s]
    M1 = np.stack(base, 1); c1, *_ = np.linalg.lstsq(M1, q - direct[k], rcond=None); r1 = np.sqrt(np.mean((M1 @ c1 - q + direct[k]) ** 2))
    M2 = np.stack([np.ones_like(s)] + base, 1); c2, *_ = np.linalg.lstsq(M2, q, rcond=None); r2 = np.sqrt(np.mean((M2 @ c2 - q) ** 2))
    A[k] = c1[0]
    print(f"  {k:6s}: limit FIXED at direct {direct[k]:.10f}: rms {r1:.1e}  a {c1[0]:+.4f} b {c1[1]:+.4f} c {c1[2]:+.4f} d {c1[3]:+.2f}"
          f"   | limit FREE: {c2[0]:.10f} (direct - fit = {direct[k]-c2[0]:+.1e}) rms {r2:.1e}")
print(f"  non-oscillating parts: a_mu/a_lam {A['mu']/A['lam']:+.4f} (tangent of the singular family -0.0430), a_y/a_lam {A['y_peak']/A['lam']:+.3f} (-2.33)")
