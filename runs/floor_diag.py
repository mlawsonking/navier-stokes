"""Where does the residual floor of the high-peak states sit, and what moves it?  floor_diag.py file.pt index [h R rc]"""
import sys, math, torch, time
from gpu_solver import ccf_sing as S, ccf_fast as F
DEV = "cuda"; sols = torch.load(sys.argv[1]); x = sols[int(sys.argv[2])]
h = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0125; R = float(sys.argv[4]) if len(sys.argv) > 4 else 6; rc = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-9
p = x["p"].to(DEV); mp = tuple(x["map"]); nj, J = x["njJ"]; n0 = p.numel() - 2 - nj * J; shape = (n0, nj, J); lam = p[0].item(); F.MAP = mp
G = S.peak_grid(lam, x["ypk"], mp, n0, h=h, R=R, device=DEV)
r = S.residual(p, G, shape); R_ = r[:-2]; y = G.y
BTh, BOm, AU, up0, wt, nvec = S.operators(p[0], p[1], G, shape); Om = (BOm @ p[2:]).abs(); i = Om.argmax(); pk = Om[i].item(); ys = y[i].item()
hw = (Om > pk / 2).nonzero().flatten(); w = (y[hw[-1]] - y[hw[0]]).item() / 2
print(f"lam {lam:.7f} mu {p[1].item():.7f} peak {pk:.2f}@{ys:.5f} HW {w:.3e} rows {y.numel()} sup {R_.abs().max().item():.2e} l2 {r.norm().item():.2e} last2 {r[-2].item():.1e} {r[-1].item():.1e}")
d = (y - ys) / w
for a, b in [(-1e9, -1e4), (-1e4, -1e3), (-1e3, -100), (-100, -30), (-30, -10), (-10, -3), (-3, -1), (-1, 1), (1, 3), (3, 10), (10, 30), (30, 100), (100, 1e3), (1e3, 1e4), (1e4, 1e9)]:
    m = (d >= a) & (d < b)
    if m.any(): print(f"  d/HW in [{a:8.0e},{b:8.0e})  rows {int(m.sum()):5d}  max|R| {R_[m].abs().max().item():.2e}  rms {R_[m].pow(2).mean().sqrt().item():.2e}")
if len(sys.argv) > 6:
    t0 = time.time(); q, hist = S.gn(p, G, shape, free=(False, True), max_iter=int(sys.argv[6]), rc=rc, rc0=rc, verbose=True, floor=1e-12)
    print("after", S.residual(q, G, shape).abs().max().item(), "mu", q[1].item(), f"t={time.time()-t0:.0f}s")
