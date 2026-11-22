"""Does the floor of a high-peak state move with n0 / map width / rows?  floor_diag2.py file.pt idx n0 epsfac h R [its]"""
import sys, math, torch, time
from gpu_solver import ccf_sing as S, ccf_fast as F
DEV = "cuda"; x = torch.load(sys.argv[1])[int(sys.argv[2])]; n0 = int(sys.argv[3]); ef = float(sys.argv[4]); h = float(sys.argv[5]); R = float(sys.argv[6]); its = int(sys.argv[7]) if len(sys.argv) > 7 else 6
p = x["p"].to(DEV); mp = tuple(x["map"]); nj, J = x["njJ"]; nold = p.numel() - 2 - nj * J; lam = p[0].item()
new = (mp[0], mp[1] * ef)
p = torch.cat([p[:2 + nold], torch.zeros(max(0, n0 - nold), dtype=S.DT, device=DEV), p[2 + nold:]]); shape = (n0, nj, J)
if ef != 1: p = S.remap0(p, mp, new, shape, DEV)
F.MAP = new; G = S.peak_grid(lam, x["ypk"], new, n0, h=h, R=R, device=DEV)
print(f"n0 {n0} eps {new[1]:.2e} rows {G.y.numel()} start sup {S.residual(p, G, shape).abs().max().item():.2e}", flush=True)
t0 = time.time(); q, hist = S.gn(p, G, shape, free=(False, True), max_iter=its, rc=1e-11, rc0=1e-9, verbose=True, floor=1e-12)
c = q[2:2 + n0].abs(); print(f"after sup {S.residual(q, G, shape).abs().max().item():.2e} mu {q[1].item():.9f} c at n/4 {c[n0//4-2:n0//4+1].max().item():.0e} n/2 {c[n0//2-2:n0//2+1].max().item():.0e} 3n/4 {c[3*n0//4-2:3*n0//4+1].max().item():.0e} last {c[-3:].max().item():.0e} t={time.time()-t0:.0f}s")
