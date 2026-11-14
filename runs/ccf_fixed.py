"""Fixed-lam solves with the fast solver (k=1 family: Omega'(0) = -2, origin row).
Usage: ccf_fixed.py n h max_iter device lam1,lam2,... [guess.pt]   -> prints, appends runs/ccf_scan.jsonl"""
import sys, json, time, torch
from gpu_solver import ccf as C, ccf_fast as F
n, h, mi, dev = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
lams = [float(x) for x in sys.argv[5].split(",")]
G = C.Grid(h=h, L=40.0, Lc=22.0, device=dev)
guess = torch.load(sys.argv[6]).to(dev).reshape(-1) if len(sys.argv) > 6 else None
out = open("runs/ccf_scan.jsonl", "a"); res = []
for lam in lams:
    c = torch.zeros(n, dtype=C.DT, device=dev)
    if guess is not None:
        m = min(n, guess.numel() - 1); c[:m] = guess[1:1 + m]
    else:
        c[0] = -1 / ((1 - 1 / (1 + lam)) / 2)
    p0 = torch.cat([torch.tensor([lam], dtype=C.DT, device=dev), c])
    t0 = time.time()
    p, hist = F.lm(p0, G, lam_free=False, max_iter=mi)
    r = F.residual(p[0], p[1:], G)
    cc = p[1:].abs()
    print(f"lam {lam:.7f} l2 {r.norm().item():.3e} sup {r.abs().max().item():.2e} origin_row {r[-1].item():+.1e} its {len(hist)} "
          f"|c| k=n/2 {cc[n//2-2:n//2+1].max().item():.1e} last {cc[-3:].max().item():.1e}  {time.time()-t0:.0f}s", flush=True)
    out.write(json.dumps(dict(case="2a-ccf-fixed-lam-fast", n_basis=n, quad_h=h, lam=lam, res_l2=r.norm().item(), res_sup=r.abs().max().item(),
                              origin_row=r[-1].item(), iters=len(hist))) + "\n")
    res.append(p)
torch.save(torch.stack(res), f"runs/ccf_fixed_n{n}_h{h}.pt")
