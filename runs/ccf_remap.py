"""Carry a solution to a new sinh map / basis size by Chebyshev interpolation of M(s) (well-conditioned),
then LM. Usage: ccf_remap.py src.pt sx0 seps out.pt n h d x0 eps max_iter free(0/1) [lam]
(sx0 = 'none' for a plain-basis source)."""
import sys, os, math, time, torch
from gpu_solver import ccf as C, ccf_fast as F
a = sys.argv
src, outp, n, h, d, x0, eps, mi, free = a[1], a[4], int(a[5]), float(a[6]), float(a[7]), float(a[8]), float(a[9]), int(a[10]), a[11] == "1"
DEV = os.environ.get("DEV", "cpu")
F.MODE.update(norm="origin", origin_row=True, d=d)
g = torch.load(src).to(DEV).reshape(-1)
# Chebyshev nodes of the NEW map, pulled back to s
j = torch.arange(n, dtype=C.DT, device=DEV)
t = torch.cos(math.pi * (j + 0.5) / n)
a1, a2 = math.asinh((1 - x0) / eps), math.asinh((1 + x0) / eps)
x = x0 + eps * torch.sinh((a1 + a2) * (t + 1) / 2 - a2)
s = ((x + 1) / 2).clamp(1e-300, 1.0)
F.MAP = None if a[2] == "none" else (float(a[2]), float(a[3]))
Mold = F.basis3(s, g.numel() - 1)[0] @ g[1:]
F.MAP = (x0, eps)
c = torch.linalg.solve(F.basis3(s, n)[0], Mold)
p = torch.cat([g[:1], c])
if len(a) > 12: p[0] = float(a[12])
G = C.Grid(h=h, L=40.0, Lc=22.0, device=DEV, n_gl=int(os.environ.get("NGL", 128)))
r0 = F.residual(p[0], p[1:], G)
t0 = time.time()
p, hist = F.lm(p, G, lam_free=free, max_iter=mi, verbose=bool(os.environ.get("VERBOSE")))
r = F.residual(p[0], p[1:], G); ops = F.operators(p[0], G, n); Om = ops[0] @ p[1:]; i = Om.abs().argmax(); cc = p[1:].abs()
print(f"n={n} h={h} map=({x0},{eps}) free={free}: start sup {r0.abs().max().item():.1e} -> lam {p[0].item():.14f} sup {r.abs().max().item():.2e} its {len(hist)} "
      f"U'(0) {(ops[4] @ p[1:]).item():.12f} peak {Om[i].abs().item():.6f}@{G.y[i].item():.4f} |c| "
      + " ".join(f"{cc[k].item():.0e}" for k in range(0, n, max(1, n // 8))) + f" last {cc[-3:].max().item():.1e}  {time.time()-t0:.0f}s", flush=True)
torch.save(p, outp)
