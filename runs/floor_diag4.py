"""Test: is the 'floor' of the high-peak states just lam predicted past a fold?  Re-solve with mu FIXED and lam FREE."""
import sys, math, torch, time
from gpu_solver import ccf_sing as S, ccf_fast as F
DEV = "cuda"; sols = torch.load(sys.argv[1])
for k in [int(a) for a in sys.argv[2:]]:
    x = sols[k]; p = x["p"].to(DEV); mp = tuple(x["map"]); nj, J = x["njJ"]; n0 = p.numel() - 2 - nj * J; shape = (n0, nj, J); F.MAP = mp
    G = S.peak_grid(p[0].item(), x["ypk"], mp, n0, h=0.0125, R=6, device=DEV)
    s0 = S.residual(p, G, shape).abs().max().item(); t0 = time.time()
    q, hist = S.gn(p, G, shape, free=(True, False), max_iter=8, rc=1e-9, rc0=1e-9, floor=1e-12)
    G2 = S.peak_grid(q[0].item(), x["ypk"], mp, n0, h=0.01, R=7.8, device=DEV, shift=0.37, widen=1.4); off = S.residual(q, G2, shape).abs().max().item()
    Om = (S.operators(q[0], q[1], G, shape)[1] @ q[2:]).abs().max().item()
    print(f"idx {k} mu {p[1].item():.8f} lam {p[0].item():.9f} -> {q[0].item():.9f} (d {q[0].item()-p[0].item():+.2e}) sup {s0:.1e} -> {hist[-1][1] if hist else s0:.1e} off {off:.1e} its {len(hist)} peak~{Om:.2f} t={time.time()-t0:.0f}s", flush=True)
    del G, G2; torch.cuda.empty_cache()
