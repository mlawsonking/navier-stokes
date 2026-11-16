"""Off-grid residual of stored ccf_fast-format branch solutions (no refit): residual of the vorticity equation on two
exponential grids the solution was not fitted on (the fits used the sinh grid). Usage: branch_offgrid_check.py file.pt"""
import sys, math, json, torch
from gpu_solver import ccf_fast as F, ccf_sing as S
DEV = "cuda"; yy = torch.exp(torch.linspace(math.log(0.05), math.log(60.0), 200001, dtype=S.DT, device=DEV))
for x in torch.load(sys.argv[1]):
    a = x["a"]; p = x["p"].to(DEV); mp = tuple(x["map"]); F.MAP = mp; n = p.numel() - 1; lam = p[0].item()
    Om = (F._basis(p[0], yy, n)[1] @ p[1:]).abs(); ypk = yy[Om.argmax()].item(); out = []
    for kw in (dict(h=0.0125, R=6), dict(h=0.01, R=7.8, shift=0.37, widen=1.4)):
        G = S.peak_grid(lam, ypk, mp, n, device=DEV, **kw); F.MAP = mp
        out.append(F.residual(p[0], p[1:], G, a)[:-2].abs().max().item()); del G; torch.cuda.empty_cache()
    print(f"a {a:+.4f} lam {lam:+.12f} peak {Om.max().item():.3f} off-grid residual {out[0]:.1e} {out[1]:.1e}", flush=True)
    open("runs/gclm_u1_branch.jsonl", "a").write(json.dumps(dict(case="gclm-u1-offgrid-check", src=sys.argv[1], a=a, lam=lam, n_basis=n, peak=Om.max().item(), res_offgrid=max(out), res_offgrid2=min(out))) + "\n")
