"""Track the non-smooth CCF family PAST a possible turning point in lam: march in mu, solve at fixed lam.
sing_track2 steps in lam and stalls where d lam / d(1/peak) -> 0 (seen below lam = 0.4546). Here the predictor is a
Lagrange extrapolation IN mu of everything (lam, coefficients in the peak-aligned map, map, peak position); the solve
is the same fixed-lam solve as before (mu free), at the predicted lam. At a fold the fixed-lam Jacobian has the family
tangent as a null direction, which the truncated SVD drops, so the iteration lands on the family near the guess.
Peak and width are measured on a local fine grid (the fixed grid of sing_track2 has about 1 point per half-width here).
Usage: sing_track3.py SRC.pt tag dmu n_steps [n0 h]   env: EPSF, ACCEPT (relative to (peak/58)^4), ROWS, DIR=-1"""
import sys, os, json, math, time, torch
from gpu_solver import ccf_sing as S, ccf_fast as F
A = sys.argv; src, tag, dmu, nsteps = A[1], A[2], float(A[3]), int(A[4])
sols = torch.load(src); nj, J = sols[-1]["njJ"]; n0 = int(A[5]) if len(A) > 5 else sols[-1]["p"].numel() - 2 - nj * J
h = float(A[6]) if len(A) > 6 else 0.0125; shape = (n0, nj, J)
DEV = "cuda"; EPSF = float(os.environ.get("EPSF", 1.0)); ACC = float(os.environ.get("ACCEPT", 3e-7)); RR = float(os.environ.get("ROWS", 6)); DIR = float(os.environ.get("DIR", -1))
yc = torch.exp(torch.linspace(math.log(0.3), math.log(60.0), 120001, dtype=S.DT, device=DEV))

def measure(p, mp):
    F.MAP = mp; al = 1 / (1 + p[0].item()); sf = lambda v: (1 + v * v) ** (-al / 2)
    om = lambda y: (S._fields(p[0], p[1], y, shape)[1] @ p[2:]).abs()
    Om = om(yc); i = Om.argmax(); y0 = yc[i].item(); pk = Om[i].item(); w = 4 * y0 * 2.5e-5
    for _ in range(6):                                   # zoom until the half-maximum set is well inside and resolved
        yl = torch.linspace(y0 - w, y0 + w, 20001, dtype=S.DT, device=DEV); Om = om(yl); i = Om.argmax()
        y0, pk = yl[i].item(), Om[i].item(); hw = (Om > pk / 2).nonzero().flatten(); a, b = yl[hw[0]].item(), yl[hw[-1]].item()
        if hw[0] > 0 and hw[-1] < 20000 and hw.numel() > 400: break
        w = 4 * (b - a) if (hw[0] > 0 and hw[-1] < 20000) else 4 * w
    return pk, y0, 2 * sf(y0) - 1, 2 * (sf(a) - sf(b)), b - a

def align(p, mp):
    pk, ypk, x0, fw, fwy = measure(p, mp); own = (x0, EPSF * fw)
    return S.remap0(p, mp, own, shape, DEV), own, ypk, pk, fwy

def padto(v):
    nold = v.numel() - 2 - nj * J
    return torch.cat([v[:2 + min(nold, n0)], torch.zeros(max(0, n0 - nold), dtype=S.DT, device=DEV), v[2 + nold:]])

states = []
NP = int(os.environ.get("NPRED", 4))
for x in sols[-NP:]:
    p, own, ypk, pk, fwy = align(padto(x["p"].to(DEV)), tuple(x["map"])); states.append((p, own, ypk))
    print(f"start lam {p[0].item():.6f} mu {p[1].item():.6f} peak {pk:.3f}@{ypk:.5f} FWHM {fwy:.3e} peak^2*HW {pk*pk*fwy/2:.4f} eps {own[1]:.3e}", flush=True)
out = open("runs/sing_track.jsonl", "a"); t0 = time.time(); step0 = dmu; done = 0; new = []
while done < nsteps:
    use = states[-NP:]; ms = [st[0][1].item() for st in use]; p, mp, ypk = states[-1]; mu_t = ms[-1] + DIR * dmu
    wts = [math.prod((mu_t - ms[k]) / (ms[i] - ms[k]) for k in range(len(use)) if k != i) for i in range(len(use))]
    guess = sum(w_ * st[0] for w_, st in zip(wts, use)); lam = guess[0].item()
    cur = (sum(w_ * st[1][0] for w_, st in zip(wts, use)), math.exp(sum(w_ * math.log(st[1][1]) for w_, st in zip(wts, use))))
    cur = (cur[0], min(max(cur[1], 0.5 * mp[1]), 2 * mp[1])); yg = math.exp(sum(w_ * math.log(st[2]) for w_, st in zip(wts, use)))
    F.MAP = cur; G = S.peak_grid(lam, yg, cur, n0, h=h, R=RR, device=DEV)
    print(f"  try mu {mu_t:.6f} lam {lam:.7f} rows {G.y.numel()} eps {cur[1]:.2e} t={time.time()-t0:.0f}s", flush=True)
    q, hist = S.gn(guess, G, shape, free=(False, True), max_iter=25, rc0=float(os.environ.get("RC0", 1e-9)))
    F.MAP = cur; sup = S.residual(q, G, shape).abs().max().item(); del G; torch.cuda.empty_cache()
    ok = sup < 1e-4 and (ms[-1] - q[1].item()) * (-DIR) > 0.2 * dmu
    if ok:
        q2, own, yq_, pk, fwy = align(q, cur); F.MAP = own; scale = max(1.0, (pk / 58) ** 4)
        G2 = S.peak_grid(lam, yq_, own, n0, h=h * 0.8, R=RR * 1.3, device=DEV, shift=0.37, widen=1.4); off = S.residual(q2, G2, shape).abs().max().item(); del G2; torch.cuda.empty_cache()
        ok = off / scale < ACC and off < 10 * sup + 1e-9
    if not ok:
        dmu /= 2; print(f"  rejected (sup {sup:.1e}, mu {q[1].item():.6f}, its {len(hist)}{'' if sup >= 1e-4 else f', off {off:.1e}' if 'off' in dir() else ''}); dmu -> {dmu:.2e}", flush=True)
        if dmu < 2e-6: print("stopping: step too small"); break
        continue
    q = q2; c = q[2:2 + n0].abs(); u0 = (1 + lam) - lam / q[1].item()
    print(f"lam {lam:.8f} mu {q[1].item():.8f} U'(0) {u0:.8f} sup {sup:.1e} OFF {off:.1e} its {len(hist)} peak {pk:.3f}@{yq_:.5f} FWHM {fwy:.3e} pk2HW {pk*pk*fwy/2:.4f} eps {own[1]:.2e} c_mid {c[n0//2-2:n0//2+1].max().item():.0e} c_last {c[-3:].max().item():.0e} t={time.time()-t0:.0f}s", flush=True)
    out.write(json.dumps(dict(case="sing-track3-" + tag, lam=lam, mu=q[1].item(), up0=u0, shape=list(shape), quad_h=h, res_sup=sup, res_offgrid=off, iters=len(hist), peak=pk, y_peak=yq_, fwhm=fwy, map=list(own))) + "\n"); out.flush()
    states.append((q, own, yq_)); states = states[-NP:]; done += 1
    new.append(dict(lam=lam, p=q.clone().cpu(), map=own, ypk=yq_, njJ=(nj, J))); torch.save(sols + new, f"runs/sing_track3_{tag}.pt")
    dmu = min(dmu * 1.5, step0)
