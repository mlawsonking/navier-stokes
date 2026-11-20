"""Shape of the non-smooth CCF family (a = -1) as its peak grows below lam_2: is the peak a symmetric pole (as on the
a-branch near a = 0) or a one-sided front with V = (1+lam) y - U vanishing at it (Huang-Tong-Wang a < 0 type, local
exponent 1/(1-a) = 1/2)?  Usage: family_end_fingerprint.py track2.pt [track2.pt ...]"""
import sys, math, torch
from gpu_solver import ccf_fast as F, ccf_sing as S
DEV = "cuda"; sols = []
for f in sys.argv[1:]: sols += torch.load(f)
sols = sorted(sols, key=lambda x: -x["lam"]); pick = sols[:: max(1, len(sols) // 7)] + [sols[-1]]
for x in pick:
    p = x["p"].to(DEV); mp = tuple(x["map"]); F.MAP = mp; nj, J = x["njJ"]; n0 = p.numel() - 2 - nj * J; shape = (n0, nj, J); lam = p[0].item()
    G = S.peak_grid(lam, x["ypk"], mp, n0, h=0.0125, R=6, device=DEV); F.MAP = mp
    BTh, BOm, AU, up0, wt, nvec = S.operators(p[0], p[1], G, shape); c = p[2:]; y = G.y
    Om, U = (BOm @ c).abs(), AU @ c; V = (1 + lam) * y - U
    i = Om.argmax(); pk = Om[i].item(); ys = y[i].item(); hw = (Om > pk / 2).nonzero().flatten()
    wl, wr = ys - y[hw[0]].item(), y[hw[-1]].item() - ys
    j = V.abs().argmin() if (V[:-1] * V[1:] < 0).any() else None; vmin = V[y > 0.2 * ys].min().item()
    sl = []
    for s in (1, -1):
        for d in (4, 16):
            j1 = (y - (ys + s * d * (wr if s > 0 else wl))).abs().argmin(); j2 = (y - (ys + s * 2 * d * (wr if s > 0 else wl))).abs().argmin()
            sl.append(math.log(Om[j2].item() / Om[j1].item()) / math.log(abs(y[j2].item() - ys) / abs(y[j1].item() - ys)))
    print(f"lam {lam:.6f} mu {p[1].item():.5f} peak {pk:8.3f} y* {ys:.4f} HW left {wl/ys:.2e} right {wr/ys:.2e} R/L {wr/wl:.2f} | V(y*)/((1+lam)y*) {V[i].item()/((1+lam)*ys):+.4f} min V/((1+lam)y*) {vmin/((1+lam)*ys):+.4f} | log-slope of |Om| outside (4w,16w) {sl[0]:+.2f} {sl[1]:+.2f} inside {sl[2]:+.2f} {sl[3]:+.2f} | peak*HWHM/y* {pk*(wl+wr)/2/ys:.4f}", flush=True)
    del G; torch.cuda.empty_cache()
