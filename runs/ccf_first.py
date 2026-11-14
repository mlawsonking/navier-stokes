"""First CCF solves: batched multi-start in lam0, small basis. Exploratory; prints only."""
import sys, time, torch
from gpu_solver import ccf as C
from gpu_solver.batch_lm import batched_lm
dev, DT = "cuda", torch.float64
n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
G = C.Grid(h=0.1, L=40.0, Lc=22.0)
lam0s = [0.6, 1.2]
P0 = []
k = torch.arange(1, n, dtype=DT, device=dev)
torch.manual_seed(0)
for l0 in lam0s:
    beta = (1 - 1 / (1 + l0)) / 2
    c = torch.cat([torch.tensor([-1 / beta], dtype=DT, device=dev), 0.05 * torch.randn(n - 1, dtype=DT, device=dev) / k ** 3])
    P0.append(torch.cat([torch.tensor([l0], dtype=DT, device=dev), c]))
f = lambda p: C.residual_vec(p, G, a=-1.0)
t0 = time.time()
p, sup, it = batched_lm(f, torch.stack(P0), max_iter=300)
torch.cuda.synchronize()
print(f"n={n} rows={G.y.numel()+2} sources={G.ys.numel()} wall {time.time()-t0:.1f}s")
for b, l0 in enumerate(lam0s):
    U, Up, Up0 = C.velocity(p[b], G)
    print(f"lam0={l0:.2f} -> lam={p[b,0].item():.13f} sup={sup[b].item():.2e} it={it[b].item()} U'(0)={Up0.item():.10f} |c_last3|={p[b,-3:].abs().max().item():.1e}")
torch.save(p, f"runs/ccf_first_n{n}.pt")
