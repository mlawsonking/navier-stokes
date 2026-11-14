"""Refine an unstable-profile candidate. Usage: ccf_u_refine.py in.pt out.pt n h d max_iter free(0/1) [lam_override]"""
import sys, time, torch
from gpu_solver import ccf as C, ccf_fast as F
inp, outp, n, h, d, mi, free = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]), sys.argv[7] == "1"
F.MODE.update(norm='origin', origin_row=True, d=d)
import os; DEV = os.environ.get("DEV", "cpu"); G = C.Grid(h=h, L=40.0, Lc=22.0, device=DEV, n_gl=int(os.environ.get("NGL", 64)))
g = torch.load(inp).to(DEV).reshape(-1)
p = torch.zeros(n + 1, dtype=C.DT, device=DEV); m = min(n + 1, g.numel()); p[:m] = g[:m]
if len(sys.argv) > 8: p[0] = float(sys.argv[8])
t0 = time.time()
p, hist = F.lm(p, G, lam_free=free, max_iter=mi)
r = F.residual(p[0], p[1:], G); ops = F.operators(p[0], G, n); Om = ops[0] @ p[1:]; i = Om.abs().argmax()
cc = p[1:].abs()
print(f"n={n} h={h} free={free}: lam {p[0].item():.14f} sup {r.abs().max().item():.2e} l2 {r.norm().item():.2e} its {len(hist)} U'(0) {(ops[4] @ p[1:]).item():.10f} "
      f"peak {Om[i].abs().item():.5f}@{G.y[i].item():.3f} |c| " + " ".join(f"{cc[j].item():.0e}" for j in range(0, n, max(1, n // 8))) + f" last {cc[-3:].max().item():.1e}  {time.time()-t0:.0f}s", flush=True)
torch.save(p, outp)
