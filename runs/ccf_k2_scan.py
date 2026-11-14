"""y^3 family, lam fixed on given values, solve coefficients. Usage: ccf_k2_scan.py n max_iter lam1,lam2,... [guess.pt]"""
import sys, json, time, torch
from gpu_solver import ccf as C
from gpu_solver.batch_lm import batched_lm
dev, DT = "cuda", torch.float64
n, mi = int(sys.argv[1]), int(sys.argv[2])
lams = torch.tensor([float(x) for x in sys.argv[3].split(",")], dtype=DT, device=dev)
G = C.Grid(h=0.1, L=40.0, Lc=22.0)
A = C.lift_matrix(n)
al = 1 / (1 + lams)
c0 = torch.zeros(lams.numel(), n + 1, dtype=DT, device=dev); c0[:, 0] = -al / (2 * al - 1)
if len(sys.argv) > 4:
    g = torch.load(sys.argv[4]); c0 = g["c"] if g["c"].shape == c0.shape else c0
f = lambda c, lam: C.residual_vec_k2(torch.cat([lam.reshape(1), c]), G, A)
t0 = time.time()
c, sup, it = batched_lm(f, c0, aux=lams, max_iter=mi, fwd=True)
torch.cuda.synchronize(); wall = time.time() - t0
r = torch.func.vmap(f)(c, lams)
print(f"n={n} wall {wall:.1f}s")
out = open("runs/ccf_scan.jsonl", "a")
for b in range(lams.numel()):
    p = C.params_k2(torch.cat([lams[b:b+1], c[b]]), A)
    Up0 = C.velocity(p, G)[2].item()
    out.write(json.dumps(dict(case="2a-ccf-k2-lam-scan", n_basis=n, lam=lams[b].item(), res_l2=r[b].norm().item(), res_sup=sup[b].item(),
                              origin_row=r[b, -1].item(), iters=it[b].item(), Up0=Up0)) + "\n")
    print(f"lam {lams[b].item():.6f} l2 {r[b].norm().item():.3e} sup {sup[b].item():.2e} origin_row {r[b,-1].item():+.2e} U'(0) {Up0:.6f} it {it[b].item()} cN_last {c[b,-3:].abs().max().item():.1e}")
torch.save(dict(lams=lams, c=c), f"runs/ccf_k2_scan_n{n}.pt")
