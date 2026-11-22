"""Residual of stored states with the old and the accurate kernel.  kernel_test.py file.pt idx [idx ...]"""
import sys, torch
from gpu_solver import ccf_sing as S, ccf_fast as F
sols = torch.load(sys.argv[1])
for k in [int(a) for a in sys.argv[2:]]:
    x = sols[k]; p = x["p"].cuda(); mp = tuple(x["map"]); nj, J = x["njJ"]; n0 = p.numel() - 2 - nj * J; shape = (n0, nj, J); out = []
    for acc in (False, True):
        S.ACCURATE_KERNEL = acc; F.MAP = mp; G = S.peak_grid(p[0].item(), x["ypk"], mp, n0, h=0.0125, R=6, device="cuda")
        r = S.residual(p, G, shape)[:-2]; i = r.abs().argmax().item(); d = (r[1:] - r[:-1])[max(i - 40, 0):i + 40]
        out.append(f"{'new' if acc else 'old'} kernel: sup {r.abs().max().item():.2e} row-to-row jump near peak rms {d.pow(2).mean().sqrt().item():.2e}"); del G; torch.cuda.empty_cache()
    print(f"peak {x.get('peak', 0):.1f}  " + "   ".join(out))
