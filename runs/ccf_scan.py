"""CCF lam-scan: lam fixed on a grid, coefficients solved (collocation + normalisation + origin row),
all grid values in one batched solve. Dips of the minimum residual locate admissible lam.
Usage: ccf_scan.py n lo hi count"""
import sys, json, time, torch
from gpu_solver import ccf as C
from gpu_solver.batch_lm import batched_lm
dev, DT = "cuda", torch.float64
n, lo, hi, cnt = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
G = C.Grid(h=0.1, L=40.0, Lc=22.0)
lams = torch.linspace(lo, hi, cnt, dtype=DT, device=dev)
f = lambda c, lam: C.residual_vec(torch.cat([lam.reshape(1), c]), G, a=-1.0)
c0 = torch.zeros(cnt, n, dtype=DT, device=dev)
c0[:, 0] = -1 / ((1 - 1 / (1 + lams)) / 2)
t0 = time.time()
c, sup, it = batched_lm(f, c0, aux=lams, max_iter=200, fwd=True)
torch.cuda.synchronize(); wall = time.time() - t0
r = torch.func.vmap(f)(c, lams)
out = open("runs/ccf_scan.jsonl", "a")
print(f"n={n} wall {wall:.1f}s")
for b in range(cnt):
    l2 = r[b].norm().item()
    out.write(json.dumps(dict(case="2a-ccf-lam-scan", n_basis=n, lam=lams[b].item(), res_l2=l2, res_sup=sup[b].item(),
                              origin_row=r[b, -1].item(), iters=it[b].item(), wall_s_batch=round(wall, 1))) + "\n")
    print(f"lam {lams[b].item():.4f} l2 {l2:.3e} sup {sup[b].item():.2e} origin_row {r[b,-1].item():+.2e} it {it[b].item()}")
torch.save(dict(lams=lams, c=c), f"runs/ccf_scan_n{n}.pt")
