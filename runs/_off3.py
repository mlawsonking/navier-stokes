import torch, math, sys
from gpu_solver import ccf_sing as S, ccf_fast as F
DEV = "cuda"; x = torch.load(sys.argv[1]); p = x["p"].to(DEV); F.MAP = tuple(x["map"]); nj, J = x["njJ"]; shape = (p.numel() - 2 - nj * J, nj, J)
xp = math.log(7.7220)
for name, h, cl in [("fit 50x", 0.00625, (xp, 0.5, 0.49)), ("20x", 0.00625, (xp, 0.5, 0.475)), ("50x shift", 0.00625, (xp + 0.0023, 0.5, 0.49)), ("50x h*.8", 0.005, (xp, 0.5, 0.49)), ("40x w.4 shift", 0.005, (xp + 0.0023, 0.4, 0.39)), ("100x w1", 0.00625, (xp, 1.0, 0.99)), ("10x", 0.00625, (xp, 0.5, 0.45))]:
    G = S.EGrid(h=h, device=DEV, cluster=cl); r = S.residual(p, G, shape)[:-2].abs(); i = r.argmax(); y = G.y
    k = r > 10 * r.median()
    print(f"{name:14s} sup {r[i].item():.1e} at y {y[i].item():.5f}; rows in y[7.6,7.85]: {int(((y>7.6)&(y<7.85)).sum())}; rows>10*median: {int(k.sum())} in y [{y[k].min().item():.4f}, {y[k].max().item():.4f}]", flush=True)
    del G; torch.cuda.empty_cache()
