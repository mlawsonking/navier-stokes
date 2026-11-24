"""Probe the last family state (peak 8386) near y*: one-sided cusp amplitudes A_L, A_R (Omega ~ -A |d|^-1/2),
their product against kappa/2, and Theta* -- input for the singular end-profile solver (gpu_solver/ccf_cusp.py).
Usage: sing_end_probe.py [SRC.pt] [index]"""
import sys, math, torch
from gpu_solver import ccf_sing as S, ccf_fast as F
src = sys.argv[1] if len(sys.argv) > 1 else "runs/sing_track4_v4.pt"; k = int(sys.argv[2]) if len(sys.argv) > 2 else -1
x = torch.load(src)[k]; nj, J = x["njJ"]; p = x["p"].to("cuda"); n0 = p.numel() - 2 - nj * J; shape = (n0, nj, J)
F.MAP = tuple(x["map"]); S.CENTER = x["center"]; lam, mu = p[0].item(), p[1].item(); ys = x["ypk"]
def ev(y):
    BTh, BOm = S._fields(p[0], p[1], y, shape)[:2]; return BTh @ p[2:], BOm @ p[2:]
th0, om0 = ev(torch.tensor([ys], dtype=S.DT, device="cuda"))
kap = lam * abs(th0.item()); print(f"lam {lam:.10f} mu {mu:.10f} peak {x['peak']:.2f} y*raw {ys:.9f} Theta* {th0.item():.9f} kappa {kap:.8f} sqrt(kappa/2) {math.sqrt(kap/2):.8f}")
for d in (1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3):
    y = torch.tensor([ys - d, ys + d], dtype=S.DT, device="cuda"); th, om = ev(y)
    aL, aR = -om[0].item() * math.sqrt(d), -om[1].item() * math.sqrt(d)
    print(f"  d {d:7.0e}  A_L {aL:.6f} A_R {aR:.6f}  A_L*A_R/(kappa/2) {aL*aR/(kap/2):.6f}  A_R/A_L {aR/aL:.6f}  (Th-Th*)/sqrt d: L {(th[0].item()-th0.item())/math.sqrt(d):+.5f} R {(th[1].item()-th0.item())/math.sqrt(d):+.5f}")
