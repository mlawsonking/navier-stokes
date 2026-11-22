"""Log-periodic (spiral) test on the non-smooth CCF family, from the ledger (no GPU).
Model with the frequency FIXED by the inner problem (w = 0.649424, no fitted frequency):
    q(sigma) = q_c + sigma [a cos(2 w ln sigma) + b sin(2 w ln sigma)] (+ c sigma^2),   sigma = 1/peak.
Data: runs/sing_track.jsonl, cases sing-track2-v*, sing-track3-y1/y2 (fixed-lam solves, peak < 70) and
sing-track4-<tags given> (pinned-peak solves). Out-of-sample test: fit on peak in [lo, hi], predict the rest.
Usage: family_spiral_fit2.py tag [tag ...]   env: LO (30), HI (100)"""
import sys, os, json, math
import numpy as np
W = 0.649424; LO = float(os.environ.get("LO", 30)); HI = float(os.environ.get("HI", 100))
rows = [json.loads(l) for l in open("runs/sing_track.jsonl")]
tags = ["sing-track4-" + t for t in sys.argv[1:]]
new = [r for r in rows if r.get("case") in tags]
pmin_new = min(r["peak"] for r in new)
old = [r for r in rows if (r.get("case", "").startswith("sing-track2-v") or r.get("case") in ("sing-track3-y1", "sing-track3-y2")) and r.get("peak", 0) < pmin_new * 0.995 and r.get("res_offgrid", 1) < 3e-7]
data = sorted(old + new, key=lambda r: r["peak"]); dd = []
for r in data:                                   # drop near-duplicates in peak (keep the later entry)
    if dd and abs(r["peak"] / dd[-1]["peak"] - 1) < 2e-3: dd[-1] = r
    else: dd.append(r)
pk = np.array([r["peak"] for r in dd]); sg = 1 / pk
print(f"{len(dd)} points, peak {pk[0]:.2f} .. {pk[-1]:.2f}; pinned-peak points from peak {pmin_new:.1f}")


def design(s, model):
    if model == "spiral": return np.stack([np.ones_like(s), s * np.cos(2 * W * np.log(s)), s * np.sin(2 * W * np.log(s))], 1)
    if model == "spiral+s2": return np.stack([np.ones_like(s), s * np.cos(2 * W * np.log(s)), s * np.sin(2 * W * np.log(s)), s * s], 1)
    if model == "poly2": return np.stack([np.ones_like(s), s, s * s], 1)
    if model == "poly3": return np.stack([np.ones_like(s), s, s * s, s ** 3], 1)


def turning(co, model, s0, s1):
    """Turning points of the fitted curve for sigma in (s1, s0), found on a fine log grid; returned as peaks."""
    s = np.exp(np.linspace(math.log(s0), math.log(s1), 400001)); q = design(s, model) @ co; d = np.diff(q)
    i = np.nonzero(d[1:] * d[:-1] < 0)[0] + 1
    return [(1 / s[k], q[k]) for k in i]


for name in ("mu", "lam", "up0", "y_peak"):
    q = np.array([r[name] for r in dd]); print(f"\n== {name}")
    fit = (pk >= LO) & (pk <= HI); test = pk > HI; allm = pk >= LO
    for model in ("spiral", "spiral+s2", "poly2", "poly3"):
        co = np.linalg.lstsq(design(sg[fit], model), q[fit], rcond=None)[0]
        rin = math.sqrt(np.mean((design(sg[fit], model) @ co - q[fit]) ** 2))
        rout = math.sqrt(np.mean((design(sg[test], model) @ co - q[test]) ** 2)) if test.any() else float("nan")
        mout = np.abs(design(sg[test], model) @ co - q[test]).max() if test.any() else float("nan")
        co2 = np.linalg.lstsq(design(sg[allm], model), q[allm], rcond=None)[0]
        rall = math.sqrt(np.mean((design(sg[allm], model) @ co2 - q[allm]) ** 2))
        tp = turning(co2, model, 1 / LO, 1e-6)
        print(f"  {model:10s} fit [{LO:.0f},{HI:.0f}] rms {rin:.1e} -> predicts peak > {HI:.0f}: rms {rout:.1e} max {mout:.1e} | all-data fit rms {rall:.1e} q_c {co2[0]:.7f} turning points (peak, value): "
              + ", ".join(f"({a:.0f}, {b:.7f})" for a, b in tp[:4]))
    # observed turning points (parabola through the three points around each extremum, in ln peak)
    for k in range(1, len(q) - 1):
        if (q[k] - q[k - 1]) * (q[k + 1] - q[k]) < 0:
            x = np.log(pk[k - 1:k + 2]); c2 = np.polyfit(x, q[k - 1:k + 2], 2); xs = -c2[1] / (2 * c2[0])
            print(f"  OBSERVED turning point: peak {math.exp(xs):.2f}  {name} = {np.polyval(c2, xs):.8f}")
