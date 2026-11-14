"""Hypothesis test: unstable CCF profiles have Omega ~ y^3 at the origin. Batched multi-start, lam free.
Usage: ccf_k2.py n max_iter [h]"""
import sys, time, torch
from gpu_solver import ccf as C
from gpu_solver.batch_lm import batched_lm
dev, DT = "cuda", torch.float64
n, mi = int(sys.argv[1]), int(sys.argv[2])
h = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
G = C.Grid(h=h, L=40.0, Lc=22.0)
A = C.lift_matrix(n)
lam0s = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90]
P0 = []
kk = torch.arange(1, n + 1, dtype=DT, device=dev)
torch.manual_seed(0)
for l0 in lam0s:
    al = 1 / (1 + l0)
    P0.append(torch.cat([torch.tensor([l0, -al / (2 * al - 1)], dtype=DT, device=dev), 0.02 * torch.randn(n, dtype=DT, device=dev) / kk ** 3]))
f = lambda q: C.residual_vec_k2(q, G, A)
t0 = time.time()
q, sup, it = batched_lm(f, torch.stack(P0), max_iter=mi, fwd=True)
torch.cuda.synchronize()
print(f"n={n} h={h} wall {time.time()-t0:.1f}s")
for b, l0 in enumerate(lam0s):
    p = C.params_k2(q[b], A)
    U, Up, Up0 = C.velocity(p, G)
    print(f"lam0={l0:.2f} -> lam={q[b,0].item():.13f} sup={sup[b].item():.2e} it={it[b].item()} U'(0)={Up0.item():.10f} |cN_last3|={q[b,-3:].abs().max().item():.1e}")
torch.save(q, f"runs/ccf_k2_n{n}.pt")
