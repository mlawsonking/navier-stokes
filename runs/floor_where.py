"""Where the residual of a sing_track4 state sits (rows binned by distance from the peak in half-widths; origin rows)."""
import sys, torch
from gpu_solver import ccf_sing as S, ccf_fast as F
x = torch.load(sys.argv[1])[int(sys.argv[2])]; p = x["p"].cuda(); mp = tuple(x["map"]); nj, J = x["njJ"]; n0 = p.numel() - 2 - nj * J; shape = (n0, nj, J)
S.CENTER = x.get("center"); F.MAP = mp; G = S.peak_grid(p[0].item(), x["ypk"], mp, n0, h=0.0125, R=6, device="cuda")
r = S.residual(p, G, shape); R_ = r[:-2]; y = G.y; w = x["fwhm"] / 2; d = (y - x["ypk"]) / w
print(f"peak {x['peak']:.1f} center {S.CENTER} sup {R_.abs().max().item():.2e} amplitude row {r[-2].item():+.1e} origin relation {r[-1].item():+.1e}")
for a, b in [(-1e12, -1e6), (-1e6, -1e4), (-1e4, -100), (-100, -10), (-10, -1), (-1, 1), (1, 10), (10, 100), (100, 1e4), (1e4, 1e6), (1e6, 1e12)]:
    m = (d >= a) & (d < b)
    if m.any():
        rr = R_[m]; jump = (rr[1:] - rr[:-1]).pow(2).mean().sqrt().item() if rr.numel() > 1 else 0
        print(f"  d/HW [{a:8.0e},{b:8.0e}) rows {int(m.sum()):5d} y {y[m].min().item():.3e}..{y[m].max().item():.3e} max {rr.abs().max().item():.2e} rms {rr.pow(2).mean().sqrt().item():.2e} row-to-row {jump:.2e}")
