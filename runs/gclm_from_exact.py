"""Continue the stable branch from the exact a = 1/2 solution in a. Usage: gclm_from_exact.py n h a_end step"""
import sys, json, torch
from gpu_solver import ccf as C, ccf_fast as F
n, h, a_end, step = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
F.MAP = None; F.MODE.update(norm="origin", origin_row=True, d=16 / 3, ymin=0.0)
G = C.Grid(h=h, L=40.0, Lc=14.0, device="cpu")
p = torch.zeros(n + 1, dtype=C.DT); p[0] = -2 / 3; p[1] = 8 / 3
a, prev = 0.5, None; sgn = 1.0 if a_end > a else -1.0
out = open("runs/gclm_branch.jsonl", "a")
while (a - a_end) * sgn <= 1e-12:
    g = p if prev is None else p + (p - prev)
    q, hist = F.lm(g, G, a=a, max_iter=300)
    r = F.residual(q[0], q[1:], G, a); sup = r.abs().max().item()
    print(f"a {a:+.4f} lam {q[0].item():.15f} 1+lam {1+q[0].item():.15f} sup {sup:.1e} its {len(hist)} |c_last3| {q[-3:].abs().max().item():.1e}", flush=True)
    out.write(json.dumps(dict(case="gclm-stable-from-exact", a=a, n_basis=n, quad_h=h, lam=q[0].item(), res_sup=sup, iters=len(hist))) + "\n")
    if sup > 1e-6: print("stopping"); break
    prev, p = p, q; a = round(a + sgn * step, 10)
