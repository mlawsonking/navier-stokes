"""Fast solver vs old solver on the stable profile, and timing."""
import time, torch
from gpu_solver import ccf as C, ccf_fast as F
p = torch.load("runs/ccf_stable_h025_n96.pt")[0]
for h in (0.1, 0.05, 0.025):
    G = C.Grid(h=h, L=40.0, Lc=22.0)
    r_old = C.residual_vec(p, G); r_new = F.residual(p[0], p[1:], G)
    t0 = time.time(); q, hist = F.lm(p, G, max_iter=50); torch.cuda.synchronize()
    print(f"h={h} old-vs-new residual diff {(r_old-r_new).abs().max().item():.1e}; LM {len(hist)} its {time.time()-t0:.2f}s lam={q[0].item():.14f} sup={hist[-1][1] if hist else float('nan'):.1e}")
