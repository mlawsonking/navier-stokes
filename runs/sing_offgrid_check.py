"""Off-grid residual of the family track: evaluate on grids the solution was not fitted on."""
import torch, math, sys
from gpu_solver import ccf_sing as S, ccf_fast as F
DEV = "cuda"; sols = torch.load(sys.argv[1])
yy = torch.exp(torch.linspace(math.log(0.3), math.log(30.), 60001, dtype=S.DT, device=DEV))
for x in sols:
    if len(sys.argv) > 2 and round(x['lam'] * 1e4) % int(sys.argv[2]): continue
    p = x["p"].to(DEV); F.MAP = tuple(x["map"]); shape = (p.numel() - 34, 8, 4)
    Om = (S._fields(p[0], p[1], yy, shape)[1] @ p[2:]).abs(); xp = math.log(yy[Om.argmax()].item())
    out = []
    for cl, h in ((None, 0.00625), ((xp, 0.5, 0.45), 0.00625), ((xp + 0.0137, 0.5, 0.48), 0.005), ((xp, 0.25, 0.2425), 0.005)):
        G = S.EGrid(h=h, device=DEV, cluster=cl); r = S.residual(p, G, shape)[:-2].abs(); out.append(f"{r.max().item():.1e}"); del G; torch.cuda.empty_cache()
    print(f"lam {x['lam']:.4f} mu {p[1].item():.10f} residual sup on [plain, fit-like 10x, shifted 25x, narrow 33x]:", out, flush=True)
