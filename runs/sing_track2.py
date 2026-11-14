"""Track the non-smooth CCF family in lam (mu free), peak-aligned predictor.
Each stored solution lives in the sinh map centred on ITS OWN peak (x0 = peak, eps = EPSF * FWHM). The secant
predictor extrapolates the map parameters, the peak position (for the clustered grid) and the coefficients
relative to the map, so a shifting/sharpening peak is carried by the map and not by the linearisation.
Usage: sing_track2.py lam_end step tag n0 nj J h   env: START=smooth-only .pt | SRC=track2 .pt, EPSF, CLUSTER, ACCEPT"""
import sys, os, json, math, time, torch
from gpu_solver import ccf_sing as S, ccf_fast as F
A = sys.argv; lam_end, step, tag = float(A[1]), float(A[2]), A[3]
n0, nj, J, h = int(A[4]), int(A[5]), int(A[6]), float(A[7]); shape = (n0, nj, J)
DEV = os.environ.get("DEV", "cuda"); EPSF = float(os.environ.get("EPSF", 1.0)); CL = float(os.environ.get("CLUSTER", 80)); ACC = float(os.environ.get("ACCEPT", 1e-7)); RR = float(os.environ.get("ROWS", 6))
yy = torch.exp(torch.linspace(math.log(0.3), math.log(60.0), 240001, dtype=S.DT, device=DEV))

def measure(p, mp):
    F.MAP = mp; al = 1 / (1 + p[0].item())
    Om = (S._fields(p[0], p[1], yy, shape)[1] @ p[2:]).abs()
    i = Om.argmax(); pk = Om[i].item(); hw = (Om > pk / 2).nonzero().flatten(); sf = lambda v: (1 + v * v) ** (-al / 2)
    return pk, yy[i].item(), 2 * sf(yy[i].item()) - 1, 2 * (sf(yy[hw[0]].item()) - sf(yy[hw[-1]].item()))

def align(p, mp):
    pk, ypk, x0, fw = measure(p, mp); own = (x0, max(EPSF * fw, float(os.environ.get("EPSMIN", 0.0004))))
    return S.remap0(p, mp, own, shape, DEV), own, ypk, pk

def padto(v, nold):
    return torch.cat([v[:2 + min(nold, n0)], torch.zeros(max(0, n0 - nold), dtype=S.DT, device=DEV), v[2 + nold:]])

states = []                                         # (p, map, ypk) of accepted solutions, each in its own map
if os.environ.get("START"):
    x = torch.load(os.environ["START"]); p = x["p"].to(DEV); nold = p.numel() - 2
    p = torch.cat([padto(p, nold), torch.zeros(nj * J, dtype=S.DT, device=DEV)])
    p, mp, ypk, _ = align(p, tuple(x["map"])); states.append((p, mp, ypk)); sols = []
else:
    sols = torch.load(os.environ["SRC"])
    for x in sols[-3:]:
        p = padto(x["p"].to(DEV), x["p"].numel() - 2 - nj * J); states.append((p, tuple(x["map"]), x["ypk"]))
lam = round(states[-1][0][0].item() - step, 10) if len(states) > 1 else round(math.floor(states[-1][0][0].item() / step) * step, 10)
if len(states) == 1 and states[-1][0][0].item() - lam < 0.2 * step: lam = round(lam - step, 10)
out = open("runs/sing_track.jsonl", "a"); t0 = time.time(); step0 = step
while lam >= lam_end - 1e-12:
    p, mp, ypk = states[-1]
    if len(states) > 1:
        # Lagrange extrapolation in lam through the last 2 or 3 accepted states: coefficients (each in its own
        # peak-aligned map), map centre, log(map width), log(peak position)
        use = states[-3:]; ls = [st[0][0].item() for st in use]
        wts = [math.prod((lam - ls[k]) / (ls[i] - ls[k]) for k in range(len(use)) if k != i) for i in range(len(use))]
        guess = sum(w_ * st[0] for w_, st in zip(wts, use))
        cur = (sum(w_ * st[1][0] for w_, st in zip(wts, use)), math.exp(sum(w_ * math.log(st[1][1]) for w_, st in zip(wts, use))))
        cur = (cur[0], min(max(cur[1], 0.5 * mp[1]), 2 * mp[1])); yg = math.exp(sum(w_ * math.log(st[2]) for w_, st in zip(wts, use)))
    else:
        guess, cur, yg = p.clone(), mp, ypk
    guess[0] = lam; F.MAP = cur
    G = S.peak_grid(lam, yg, cur, n0, h=h, R=RR, device=DEV)
    q, hist = S.gn(guess, G, shape, free=(False, True), max_iter=int(os.environ.get("MAXIT", 25)))
    F.MAP = cur; sup = S.residual(q, G, shape).abs().max().item(); del G; torch.cuda.empty_cache()
    if not sup < ACC:
        lam = round(lam + step, 10); step = step / 2; lam = round(lam - step, 10)
        print(f"  rejected (sup {sup:.1e}, its {len(hist)}); step -> {step}", flush=True)
        if step < 1e-5: print("stopping: step too small"); break
        continue
    q, own, yq_, pk = align(q, cur); F.MAP = own
    G2 = S.peak_grid(lam, yq_, own, n0, h=h * 0.8, R=RR * 1.3, device=DEV, shift=0.37, widen=1.4); off = S.residual(q, G2, shape).abs().max().item(); del G2; torch.cuda.empty_cache()
    c = q[2:2 + n0].abs(); cs = q[2 + n0:].abs(); u0 = (1 + lam) - lam / q[1].item()
    print(f"lam {lam:.6f} mu {q[1].item():.10f} U'(0) {u0:.9f} sup {sup:.1e} OFF {off:.1e} its {len(hist)} peak {pk:.4f}@{yq_:.4f} eps {own[1]:.4f} c_mid {c[n0//2-2:n0//2+1].max().item():.0e} c_last {c[-3:].max().item():.0e} sing {cs.max().item() if cs.numel() else 0:.1e} t={time.time()-t0:.0f}s", flush=True)
    out.write(json.dumps(dict(case="sing-track2-" + tag, lam=lam, mu=q[1].item(), up0=u0, shape=list(shape), quad_h=h, res_sup=sup, res_offgrid=off, iters=len(hist), peak=pk, y_peak=yq_, map=list(own))) + "\n"); out.flush()
    states.append((q, own, yq_)); states = states[-3:]
    sols.append(dict(lam=lam, p=q.clone().cpu(), map=own, ypk=yq_, njJ=(nj, J))); torch.save(sols, f"runs/sing_track2_{tag}.pt")
    step = min(step * 1.5, step0); lam = round(lam - step, 10)
