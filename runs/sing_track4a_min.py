"""mu along the family at fixed a from the fixed-peak marches of sing_track4a (runs/sing_track4a.jsonl):
table mu(peak; a), and the minimum of mu over the sampled peaks by a parabola in ln peak through the three
lowest neighbouring points (only if the minimum is bracketed).   Usage: sing_track4a_min.py tag1 tag2 ...  (e.g. m153 m251 m317 m400)"""
import sys, json, math, numpy as np
tags = sys.argv[1:]
rows = [json.loads(l) for l in open("runs/sing_track4a.jsonl")]
dat = {}
for r in rows:
    t = r["case"].replace("sing-track4a-", "")
    if t in tags: dat[(t, round(r["a"], 5))] = r
avals = sorted({a for (_, a) in dat}, reverse=True)
print("   a      " + "  ".join(f"{t:>26s}" for t in tags) + "   mu_min (parabola)  at peak")
for a in sorted(avals, reverse=False):
    pts = [(dat[(t, a)]["peak"], dat[(t, a)]["mu"], dat[(t, a)]["lam"], dat[(t, a)]["res_offgrid"]) for t in tags if (t, a) in dat]
    cells = "  ".join(f"{dat[(t, a)]['mu']:.9f} ({dat[(t, a)]['res_offgrid']:.0e})" if (t, a) in dat else " " * 26 for t in tags)
    best = ""
    if len(pts) >= 3:
        pts.sort(); lp = np.log([p[0] for p in pts]); m = np.array([p[1] for p in pts]); i = int(np.argmin(m))
        if 0 < i < len(m) - 1:
            c = np.polyfit(lp[i - 1:i + 2], m[i - 1:i + 2], 2); x = -c[1] / (2 * c[0]); best = f"   {np.polyval(c, x):.9f}  {math.exp(x):8.1f}"
        else:
            best = f"   (not bracketed: lowest at peak {pts[i][0]:.0f}: {m[i]:.9f})"
    print(f"{a:+.4f}  {cells}{best}")
print("lam at each a:")
for a in sorted(avals):
    print(f"{a:+.4f}  " + "  ".join(f"{dat[(t, a)]['lam']:.9f}" if (t, a) in dat else " " * 11 for t in tags))
