"""Confirm a profile pinned with the singular blocks present is SMOOTH: refit with N_0 only (J = 0).
Start: N_0 <- N_0 + sum_j (1-s)^(j/2) N_j sampled at the mapped Chebyshev nodes (mu = 2).
Usage: sing_to_smooth.py in.pt n0 h out.pt"""
import sys, os, math, json, time, torch
from gpu_solver import ccf_sing as S, ccf_fast as F
DEV = "cuda"; x = torch.load(sys.argv[1]); n0, h, outp = int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
p = x["p"].to(DEV); cur = tuple(x["map"]); nj, J = x["njJ"]; nold = p.numel() - 2 - nj * J
j = torch.arange(n0, dtype=S.DT, device=DEV); t = torch.cos(math.pi * (j + 0.5) / n0)
x0, eps = cur; b1, b2 = math.asinh((1 - x0) / eps), math.asinh((1 + x0) / eps)
s = (x0 + eps * torch.sinh((b1 + b2) * (t + 1) / 2 - b2) + 1) / 2
F.MAP = cur
E, _ = S.E2(s, 1 - s, torch.tensor(2.0, dtype=S.DT, device=DEV), (nold, nj, J))
Ntot = (E @ p[2:]) / (1 - s)
c0 = torch.linalg.solve(F.basis3(s, n0)[0], Ntot)
q = torch.cat([p[:1], torch.tensor([2.0], dtype=S.DT, device=DEV), c0]); shape = (n0, 1, 0)
yy = torch.exp(torch.linspace(math.log(0.3), math.log(60.0), 120001, dtype=S.DT, device=DEV))
Om = (S._fields(q[0], q[1], yy, shape)[1] @ q[2:]).abs(); ypk = yy[Om.argmax()].item()
G = S.EGrid(h=h, device=DEV, cluster=("box", math.log(ypk), float(os.environ.get("CC", 0.15)), float(os.environ.get("CD", 40))))
r = S.residual(q, G, shape); print("start residual sup", r.abs().max().item(), "tail coef", c0[-8:].abs().max().item(), flush=True)
t0 = time.time(); q, hist = S.lm(q, G, shape, free=(True, False), max_iter=80, verbose=True)
r = S.residual(q, G, shape); c = q[2:].abs(); sup_on = r.abs().max().item()
del G; torch.cuda.empty_cache(); G2 = S.EGrid(h=h * 0.8, device=DEV, cluster=("box", math.log(ypk) + 0.003, 0.1, 60.0)); off = S.residual(q, G2, shape).abs().max().item()
Om = (S._fields(q[0], q[1], yy, shape)[1] @ q[2:]).abs()
print(f"SMOOTH-ONLY n0 {n0} h {h} lam {q[0].item():.14f} sup {sup_on:.1e} OFF {off:.1e} its {len(hist)} peak {Om.max().item():.5f}@{yy[Om.argmax()].item():.4f} coef", " ".join(f"{c[k:k+n0//8].max().item():.0e}" for k in range(0, n0, n0 // 8)), f"{time.time()-t0:.0f}s")
torch.save(dict(lam=q[0].item(), p=q.cpu(), map=cur, njJ=(1, 0)), outp)
open("runs/ledger.jsonl", "a").write(json.dumps(dict(case="ccf-smooth-only", n0=n0, quad_h=h, lam=q[0].item(), res_sup=sup_on, res_offgrid=off, map=list(cur))) + "\n")
