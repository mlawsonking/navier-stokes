import torch, math, sys
from gpu_solver import ccf_sing as S, ccf_fast as F
DEV = "cuda"; x = torch.load(sys.argv[1]); p = x["p"].to(DEV); F.MAP = tuple(x["map"]); nj, J = x["njJ"]; shape = (p.numel() - 2 - nj * J, nj, J)
xp = math.log(7.7220)
for name, h, cl in [("box c.15 D40", 0.00625, ("box", xp, 0.15, 40.)), ("box c.1 D60 shift", 0.00625, ("box", xp + 0.003, 0.1, 60.)), ("box c.3 D20", 0.00625, ("box", xp, 0.3, 20.)), ("box c.15 D40 h*.8", 0.005, ("box", xp, 0.15, 40.))]:
    G = S.EGrid(h=h, device=DEV, cluster=cl); r = S.residual(p, G, shape)[:-2].abs(); i = r.argmax(); y = G.y
    print(f"{name:18s} rows {y.numel()} y-range {y[0].item():.1e}..{y[-1].item():.1e} sup {r[i].item():.1e} at y {y[i].item():.5f}; rows in y[7.6,7.85]: {int(((y>7.6)&(y<7.85)).sum())}", flush=True)
    del G; torch.cuda.empty_cache()
