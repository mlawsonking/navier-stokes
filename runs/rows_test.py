"""Which rows carry the residual: collocation, amplitude row, origin relation.  rows_test.py file.pt idx... (env ROWS, H)"""
import sys, os, torch
from gpu_solver import ccf_sing as S, ccf_fast as F
sols = torch.load(sys.argv[1]); R = float(os.environ.get("ROWS", 6)); h = float(os.environ.get("H", 0.0125))
for k in [int(a) for a in sys.argv[2:]]:
    x = sols[k]; p = x["p"].cuda(); mp = tuple(x["map"]); nj, J = x["njJ"]; n0 = p.numel() - 2 - nj * J; shape = (n0, nj, J); F.MAP = mp
    G = S.peak_grid(p[0].item(), x["ypk"], mp, n0, h=h, R=R, device="cuda"); r = S.residual(p, G, shape)
    lo = G.y < 1e-2
    print(f"peak {x.get('peak', 0):.1f} rows {G.y.numel()} colloc sup {r[:-2].abs().max().item():.2e} (y<0.01: {r[:-2][lo].abs().max().item():.2e}) amplitude row {r[-2].item():+.2e} origin relation {r[-1].item():+.2e}"); del G; torch.cuda.empty_cache()
