"""Fit d mu/da and d lam/da at fixed peak (runs/gcusp_dmuda.jsonl, mode peak, cut 1e-11) to the form expected from
q(peak, a) = q_c(a) + sigma^|a| [A(a) cos(w(a)(1-a) ln peak) + B(a) sin(...)]:
   dq/da = dq_c/da + sigma [(c1 + c2 ln sigma) cos(2 w ln sigma) + (c3 + c4 ln sigma) sin(2 w ln sigma)] (+ c5 sigma^2)
(the ln sigma terms come from d/da of the exponent and the frequency).   Usage: gcusp_fit.py [peak_min ...]"""
import sys, json, numpy as np
W2 = 2 * 0.649424
rows = [json.loads(l) for l in open("runs/gcusp_dmuda.jsonl")]
rows = [r for r in rows if r["mode"] == "peak" and r["peak"] > 5]
d = {}
for r in rows: d[round(r["peak"], 3)] = r            # later entries win
pk = np.array(sorted(d)); dl = np.array([d[p]["res"]["1e-11"][0] for p in pk]); dm = np.array([d[p]["res"]["1e-11"][1] for p in pk])
for p, a, b in zip(pk, dl, dm): print(f"  peak {p:9.2f}  dlam/da {a:+.6f}  dmu/da {b:+.6f}")
for pmin in [float(v) for v in sys.argv[1:]] or [8.0, 20.0, 50.0]:
    m = pk >= pmin; s = 1 / pk[m]; L = np.log(s); C, S_ = np.cos(W2 * L), np.sin(W2 * L)
    for name, y in (("dmu/da", dm[m]), ("dlam/da", dl[m])):
        for lab, cols in (("const", [np.ones_like(s)]), ("const+sigma", [np.ones_like(s), s]),
                          ("spiral", [np.ones_like(s), s * C, s * S_]), ("spiral+log", [np.ones_like(s), s * C, s * S_, s * L * C, s * L * S_]),
                          ("spiral+log+s2", [np.ones_like(s), s * C, s * S_, s * L * C, s * L * S_, s * s])):
            M = np.stack(cols, 1)
            if M.shape[0] <= M.shape[1] + 1: continue
            c, *_ = np.linalg.lstsq(M, y, rcond=None); print(f"peak>={pmin:6.0f} n {m.sum():2d} {name:8s} {lab:14s} limit {c[0]:+.6f}  rms {np.sqrt(np.mean((M @ c - y) ** 2)):.1e}")
