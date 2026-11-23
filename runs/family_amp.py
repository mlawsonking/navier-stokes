"""Log-periodic amplitude of each quantity: q = q_c + sigma (a cos th + b sin th) + c2 sigma^2 (+ c3 sigma^3), th = 2 w ln sigma,
w = 0.649424 fixed. Reports R = sqrt(a^2+b^2), phase, c2, and the same with the oscillation REMOVED (pure polynomial in sigma)
to see whether the oscillating term is needed at all.  Usage: family_amp.py lo tag [tag ...]"""
import sys, json, math
import numpy as np
W = 0.649424; lo = float(sys.argv[1]); tags = ["sing-track4-" + t for t in sys.argv[2:]]
rows = sorted([json.loads(l) for l in open("runs/sing_track.jsonl") if '"sing-track4-' in l], key=lambda r: r["peak"])
rows = [r for r in rows if r["case"] in tags]; dd = []
for r in rows:
    if dd and abs(r["peak"] / dd[-1]["peak"] - 1) < 2e-3: dd[-1] = r if r["shape"][0] >= dd[-1]["shape"][0] else dd[-1]
    else: dd.append(r)
dd = [r for r in dd if r["peak"] >= lo and not (r["shape"][0] < 1536 and r["peak"] > 1100)]
pk = np.array([r["peak"] for r in dd]); s = 1 / pk; th = 2 * W * np.log(s); print(f"{len(dd)} points, peak {pk[0]:.0f} .. {pk[-1]:.0f} (n0 = 1024 points above peak 1100 dropped)")
for name in ("mu", "lam", "y_peak", "up0"):
    q = np.array([r[name] for r in dd])
    for label, cols in (("osc + s^2      ", [np.ones_like(s), s * np.cos(th), s * np.sin(th), s * s]), ("osc + s^2 + s^3", [np.ones_like(s), s * np.cos(th), s * np.sin(th), s * s, s ** 3]),
                        ("s + s^2 + s^3  ", [np.ones_like(s), s, s * s, s ** 3]), ("s + .. + s^4   ", [np.ones_like(s), s, s * s, s ** 3, s ** 4])):
        A = np.stack(cols, 1); co = np.linalg.lstsq(A, q, rcond=None)[0]; rms = math.sqrt(np.mean((A @ co - q) ** 2))
        extra = f"R {math.hypot(co[1], co[2]):.4f} phase {math.atan2(co[2], co[1]):+.3f} c2 {co[3]:+.3f}" if label.startswith("osc") else f"c1 {co[1]:+.4f} c2 {co[2]:+.3f}"
        print(f"  {name:6s} {label} q_c {co[0]:.8f} rms {rms:.1e}  {extra}")
