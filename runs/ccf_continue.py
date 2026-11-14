"""Continue LM (lam free) from saved parameter vectors. Usage: ccf_continue.py in.pt out.pt max_iter [n_new]"""
import sys, time, torch
from gpu_solver import ccf as C
from gpu_solver.batch_lm import batched_lm
DT = torch.float64
p = torch.load(sys.argv[1]); p = p if p.dim() == 2 else p[None]
if len(sys.argv) > 4:
    n_new = int(sys.argv[4]); p = torch.cat([p, torch.zeros(p.shape[0], n_new + 1 - p.shape[1], dtype=DT, device=p.device)], 1)
import os; G = C.Grid(h=float(os.environ.get("H", 0.1)), L=float(os.environ.get("L", 40.0)), Lc=float(os.environ.get("LC", 22.0)), n_gl=int(os.environ.get("NGL", 64)))
f = lambda q: C.residual_vec(q, G, a=-1.0)
t0 = time.time()
p, sup, it = batched_lm(f, p, max_iter=int(sys.argv[3]), fwd=True)
torch.cuda.synchronize()
for b in range(p.shape[0]):
    U, Up, Up0 = C.velocity(p[b], G)
    print(f"lam={p[b,0].item():.14f} sup={sup[b].item():.2e} it={it[b].item()} U'(0)={Up0.item():.12f} |c_last3|={p[b,-3:].abs().max().item():.1e}  wall {time.time()-t0:.0f}s")
torch.save(p, sys.argv[2])
