"""Does the first unstable branch near its end look like the Huang-Tong-Wang singular profile?
Fingerprints of that profile (front at X = 1): (i) V = (1+lam) y + a U vanishes at the front; (ii) Omega ~ (X-1)^(-mu),
mu = 1/(1-a), just outside the front and Omega -> 0 inside (one-sided); (iii) U' ~ |X-1|^(-mu) on both sides.
Usage: branch_end_fingerprint.py file.pt [index ...]"""
import sys, math, torch
from gpu_solver import ccf_fast as F, gclm_log as L, ccf_sing as S
DEV = "cuda"; sols = torch.load(sys.argv[1]); idx = [int(v) for v in sys.argv[2:]] or [-1]
for k in idx:
    x = sols[k]; a = x["a"]; p = x["p"].to(DEV); mp = tuple(x["map"]); F.MAP = mp; n = p.numel() - 2; lam = p[0].item(); yp = x["ypk"]
    G = S.peak_grid(lam, yp, mp, n, h=0.0125, R=6, device=DEV)
    BOm, BOmy, AU, AUp, up0, q1, wt = L.operators(p[0], G, n); z = p[1:]
    y = G.y; Om, U, Up = BOm @ z, AU @ z, AUp @ z; V = (1 + lam) * y + a * U
    i = Om.abs().argmax(); pk = Om[i].abs().item(); ys = y[i].item(); hw = (Om.abs() > pk / 2).nonzero().flatten()
    wl, wr = ys - y[hw[0]].item(), y[hw[-1]].item() - ys
    sg = (V[:-1] * V[1:] < 0).nonzero().flatten(); zeros = [round(y[j].item() / ys, 5) for j in sg]
    print(f"a {a:+.5f} lam {lam:+.6f} mu=1/(1-a) {1/(1-a):.4f} peak {pk:.2f} at y* {ys:.5f}; half-widths left {wl/ys:.2e} right {wr/ys:.2e} (ratio R/L {wr/wl:.2f})")
    print(f"  V at peak {V[i].item():+.4e} (scale (1+lam) y* = {(1+lam)*ys:.3e}); zeros of V at y/y* = {zeros}; U' at peak {Up[i].item():+.3f}, max U' {Up.max().item():.2f} at y/y* {y[Up.argmax()].item()/ys:.5f}, min U' {Up.min().item():.2f} at {y[Up.argmin()].item()/ys:.5f}")
    for side, s in (("outside", 1), ("inside", -1)):
        out = []
        for d in (0.003, 0.01, 0.03, 0.1, 0.3):
            j1 = (y - ys * (1 + s * d)).abs().argmin(); j2 = (y - ys * (1 + s * d * 1.5)).abs().argmin()
            d1, d2 = abs(y[j1].item() / ys - 1), abs(y[j2].item() / ys - 1)
            sl = math.log(abs(Om[j2].item() / Om[j1].item())) / math.log(d2 / d1); slu = math.log(abs(Up[j2].item() / Up[j1].item())) / math.log(d2 / d1)
            out.append(f"d={d}: |Om| {abs(Om[j1].item()):.3g} slope {sl:+.2f} U' {Up[j1].item():+.3g} slope {slu:+.2f}")
        print(f"  {side}: " + " | ".join(out))
    del G
