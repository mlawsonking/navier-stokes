"""Re-converge stored family states at fixed lam (mu free) to the rounding floor and report on = off.
sing_track3 with a relative ACCEPT let states through after a single Gauss-Newton step above peak 75; this finishes them.
Usage: sing_polish.py in.pt out.pt peak_min every"""
import sys, math, json, torch
from gpu_solver import ccf_sing as S, ccf_fast as F
src, dst, pmin, every = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]); DEV = "cuda"
sols = torch.load(src); out = []; k = 0
for x in sols:
    p = x["p"].to(DEV); nj, J = x["njJ"]; n0 = p.numel() - 2 - nj * J; sh = (n0, nj, J); mp = tuple(x["map"]); lam = p[0].item(); F.MAP = mp
    y = torch.linspace(x["ypk"] - 0.01, x["ypk"] + 0.01, 20001, dtype=S.DT, device=DEV); pk0 = (S._fields(p[0], p[1], y, sh)[1] @ p[2:]).abs().max().item()
    if pk0 < pmin: continue
    k += 1
    if k % every: continue
    G = S.peak_grid(lam, x["ypk"], mp, n0, h=0.0125, R=6, device=DEV); s0 = S.residual(p, G, sh).abs().max().item()
    q, hist = S.gn(p, G, sh, free=(False, True), max_iter=30, rc0=1e-9, tol=1e-12, floor=1e-12)
    sup = S.residual(q, G, sh).abs().max().item(); del G; torch.cuda.empty_cache()
    Om = (S._fields(q[0], q[1], y, sh)[1] @ q[2:]).abs(); i = Om.argmax(); pk = Om[i].item(); ys = y[i].item(); hw = (Om > pk / 2).nonzero().flatten(); fw = (y[hw[-1]] - y[hw[0]]).item()
    th = abs((S._fields(q[0], q[1], y[i:i+1], sh)[0] @ q[2:]).item())
    G2 = S.peak_grid(lam, ys, mp, n0, h=0.01, R=7.8, device=DEV, shift=0.37, widen=1.4); off = S.residual(q, G2, sh).abs().max().item(); del G2; torch.cuda.empty_cache()
    print(f"lam {lam:.8f} mu {x['p'][1].item():.8f} -> {q[1].item():.8f} peak {pk0:.3f} -> {pk:.3f}@{ys:.5f} sup {s0:.1e} -> {sup:.1e} OFF {off:.1e} its {len(hist)} pk2HW/kappa {pk*pk*fw/2/(lam*th):.5f}", flush=True)
    out.append(dict(lam=lam, p=q.cpu(), map=mp, ypk=ys, njJ=(nj, J), off=off)); torch.save(out, dst)
    open("runs/sing_track.jsonl", "a").write(json.dumps(dict(case="sing-polish", lam=lam, mu=q[1].item(), peak=pk, y_peak=ys, fwhm=fw, kappa=lam * th, res_sup=sup, res_offgrid=off, iters=len(hist))) + "\n")
