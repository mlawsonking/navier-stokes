"""CCF lam-scan started from a given coefficient vector (the 'unstable valley' guess).
Usage: ccf_scan2.py guess.pt row lo hi count max_iter tag"""
import sys, json, time, torch
from gpu_solver import ccf as C
from gpu_solver.batch_lm import batched_lm
dev, DT = "cuda", torch.float64
p = torch.load(sys.argv[1]); p = p[int(sys.argv[2])] if p.dim() == 2 else p
lo, hi, cnt, mi, tag = float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
n = p.numel() - 1
G = C.Grid(h=0.1, L=40.0, Lc=22.0)
lams = torch.linspace(lo, hi, cnt, dtype=DT, device=dev)
import os
if os.environ.get("LAMS"): lams = torch.tensor([float(x) for x in os.environ["LAMS"].split(",")], dtype=DT, device=dev); cnt = lams.numel()
f = lambda c, lam: C.residual_vec(torch.cat([lam.reshape(1), c]), G, a=-1.0)
c0 = p[1:][None].repeat(cnt, 1)
t0 = time.time()
c, sup, it = batched_lm(f, c0, aux=lams, max_iter=mi, fwd=True)
torch.cuda.synchronize(); wall = time.time() - t0
r = torch.func.vmap(f)(c, lams)
out = open("runs/ccf_scan.jsonl", "a")
print(f"n={n} wall {wall:.1f}s", flush=True)
for b in range(cnt):
    l2 = r[b].norm().item()
    out.write(json.dumps(dict(case="2a-ccf-lam-scan-" + tag, n_basis=n, lam=lams[b].item(), res_l2=l2, res_sup=sup[b].item(),
                              origin_row=r[b, -1].item(), iters=it[b].item(), wall_s_batch=round(wall, 1))) + "\n")
    print(f"lam {lams[b].item():.5f} l2 {l2:.3e} sup {sup[b].item():.2e} origin_row {r[b,-1].item():+.2e} it {it[b].item()} c_last {c[b,-3:].abs().max().item():.1e}")
torch.save(dict(lams=lams, c=c), f"runs/ccf_scan_{tag}.pt")
