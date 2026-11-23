"""Non-smooth CCF family at large peak: solve with the PEAK HEIGHT pinned and BOTH lam and mu free.
Why (runs/floor_diag*.py, 2026-11-21): the 'floor' of sing_track3 above peak 75 was not a resolution limit. The
residual of those states lies in the range of the Jacobian (|P r|/|r| = 0.9999); the fixed-lam solve failed because
(i) the extrapolated lam was off by ~5e-6, which at fixed lam needs a several-percent move along the family, and
(ii) with cutoff 1e-9 the truncated SVD kept directions (relative singular values 1e-5 .. 1e-9) whose steps are far
outside the linear range when the peak is sharp. Here: extra row Omega(y_g) = -peak_target (y_g = predicted peak
position), unknowns [lam, mu, c]; lam/mu columns by 4th-order differences (the 2nd-order 1e-6 column is 7e-4 off:
changing lam at fixed s moves the peak by 3 percent of its width); each iteration tries a ladder of cutoffs and
keeps the best actual residual.
Usage: sing_track4.py polish SRC.pt tag peak_min [every]      re-converge stored states at their own peak height
       sing_track4.py march  SRC.pt tag ratio n_steps         continue: peak_target = ratio * last peak
env: ACCEPT (absolute off-grid sup, default 3e-7), NPRED, ROWS, EPSF"""
import sys, os, json, math, time, torch
from gpu_solver import ccf_sing as S, ccf_fast as F
A = sys.argv; mode, src, tag = A[1], A[2], A[3]
sols = torch.load(src); nj, J = sols[-1]["njJ"]; n0 = int(os.environ.get("N0", sols[-1]["p"].numel() - 2 - nj * J)); shape = (n0, nj, J); h = 0.0125
DEV = "cuda"; EPSF = float(os.environ.get("EPSF", 1.0)); ACC = float(os.environ.get("ACCEPT", 3e-7)); RR = float(os.environ.get("ROWS", 6)); NP = int(os.environ.get("NPRED", 4))
yc = torch.exp(torch.linspace(math.log(0.3), math.log(60.0), 120001, dtype=S.DT, device=DEV))


def om_at(p, y):
    if y.numel() <= 8192: return S._fields(p[0], p[1], y, shape)[1] @ p[2:]
    return torch.cat([S._fields(p[0], p[1], yy, shape)[1] @ p[2:] for yy in y.split(8192)])   # chunked: keeps GPU memory for the solve


def padto(v):
    nold = v.numel() - 2 - nj * J
    return torch.cat([v[:2 + min(nold, n0)], torch.zeros(max(0, n0 - nold), dtype=S.DT, device=DEV), v[2 + nold:]])


def measure(p, mp):
    F.MAP = mp; al = 1 / (1 + p[0].item()); sf = lambda v: (1 + v * v) ** (-al / 2)
    Om = om_at(p, yc).abs(); i = Om.argmax(); y0 = yc[i].item(); pk = Om[i].item(); w = 4 * y0 * 2.5e-5
    for _ in range(8):
        yl = torch.linspace(y0 - w, y0 + w, 20001, dtype=S.DT, device=DEV); Om = om_at(p, yl).abs(); i = Om.argmax()
        y0, pk = yl[i].item(), Om[i].item(); hw = (Om > pk / 2).nonzero().flatten(); a, b = yl[hw[0]].item(), yl[hw[-1]].item()
        if hw[0] > 0 and hw[-1] < 20000:           # half-maximum crossings by linear interpolation (was: nearest grid point, 0.3 percent)
            i0, i1 = hw[0].item(), hw[-1].item(); a = yl[i0 - 1].item() + (yl[i0] - yl[i0 - 1]).item() * (pk / 2 - Om[i0 - 1].item()) / (Om[i0] - Om[i0 - 1]).item()
            b = yl[i1].item() + (yl[i1 + 1] - yl[i1]).item() * (Om[i1].item() - pk / 2) / (Om[i1] - Om[i1 + 1]).item()
        if hw[0] > 0 and hw[-1] < 20000 and hw.numel() > 400: break
        w = 4 * (b - a) if (hw[0] > 0 and hw[-1] < 20000) else 4 * w
    yl = torch.linspace(y0 - (b - a) / 50, y0 + (b - a) / 50, 20001, dtype=S.DT, device=DEV); Om = om_at(p, yl).abs(); i = Om.argmax()
    torch.cuda.empty_cache()
    return Om[i].item(), yl[i].item(), 2 * sf(y0) - 1, 2 * (sf(a) - sf(b)), b - a


def align(p, mp):
    pk, ypk, x0, fw, fwy = measure(p, mp); own = (x0, EPSF * fw)
    return S.remap0(p, mp, own, shape, DEV), own, ypk, pk, fwy


def solve(guess, cur, yg, target, fw, max_iter=14, tol=2e-9, verbose=True):
    # Rows: collocation, origin relation, Omega(yg) = -target, Omega(yg + dl) = Omega(yg - dl) (the peak IS at yg).
    # The amplitude row of ccf_sing (Theta ~ -(D/2)|y|^mu) is DROPPED: it fixes the scaling y -> a y, Theta -> a Theta,
    # and correcting it by 1e-6 means sliding the peak by a full width (seen as an amplitude-row residual growing from
    # 5e-12 at peak 70 to 2.7e-6 at peak 199). The scale is fixed by the peak position instead; `standard` below
    # gives the factor a that restores the usual normalisation exactly (lam, mu, peak height do not depend on it).
    dl = 0.05 * fw / 2
    # the map centre FOLLOWS lam so that the basis stays centred on y = yg: at fixed x0 a change of lam slides the
    # whole N_0 block across the peak (d ln y / d lam = ln y/(1+lam) at fixed s: 5e-6 in lam = 13 percent of the
    # half-width at peak 134), which made the lam direction violently nonlinear
    mapof = lambda lam: (2 * (1 + yg * yg) ** (-0.5 / (1 + float(lam))) - 1, cur[1])
    guess = S.remap0(guess, cur, mapof(guess[0]), shape, DEV); cur0 = cur
    S.CENTER = yg            # stays set for the measurement and the off-grid check of this state
    F.MAP = cur0; G = S.peak_grid(guess[0].item(), yg, cur0, n0, h=h, R=RR, device=DEV); ygt = torch.tensor([yg, yg + dl, yg - dl], dtype=S.DT, device=DEV)

    def resid(p):
        F.MAP = mapof(p[0]); r = S.residual(p, G, shape); o = om_at(p, ygt) / target
        return torch.cat([r[:-2], r[-1:], (o[0] + 1).reshape(1), (100 * (o[1] - o[2])).reshape(1)])

    def resjac(p):
        F.MAP = mapof(p[0]); r, Jc = S.residual(p, G, shape, with_jac=True); B = S._fields(p[0], p[1], ygt, shape)[1] / target
        pin_row = torch.stack([B[0], 100 * (B[1] - B[2])]); o = B @ p[2:]
        r = torch.cat([r[:-2], r[-1:], (o[0] + 1).reshape(1), (100 * (o[1] - o[2])).reshape(1)]); Jc = torch.cat([Jc[:-2], Jc[-1:]]); cols = []
        for k in (0, 1):
            e = torch.zeros_like(p); e[k] = 2e-7
            cols.append(((8 * (resid(p + e) - resid(p - e)) - (resid(p + 2 * e) - resid(p - 2 * e))) / (12 * 2e-7))[:, None])
        return r, torch.cat(cols + [torch.cat([Jc, pin_row])], 1)

    tq = time.time(); p = guess.clone(); r, Jm = resjac(p); print(f'    resjac {time.time()-tq:.0f}s rows {G.y.numel()}', flush=True); hist = [r[:-2].abs().max().item()]; stall = 0
    for it in range(max_iter):
        d = Jm.norm(dim=0).clamp_min(1e-300); U, sv, Vh = torch.linalg.svd((Jm / d).cpu(), full_matrices=False); co = U.T @ -r.cpu(); best = None
        for cut in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10):
            k = sv > cut * sv[0]; st = (Vh[k].T @ (co[k] / sv[k])).to(DEV) / d
            for t in (1.0,):
                nn = resid(p + t * st).norm().item()
                if math.isfinite(nn) and (best is None or nn < best[0]): best = (nn, cut, t, st)
        if best is None or best[0] >= r.norm().item():
            k = sv > 1e-4 * sv[0]; st = (Vh[k].T @ (co[k] / sv[k])).to(DEV) / d; t = 0.25; best = None
            while t > 1e-3:
                nn = resid(p + t * st).norm().item()
                if nn < r.norm().item(): best = (nn, 1e-4, t, st); break
                t /= 2
            if best is None: break
        gain = 1 - best[0] / r.norm().item(); p = p + best[2] * best[3]; r, Jm = resjac(p); hist.append(r[:-2].abs().max().item())
        if verbose: print(f"    gn {it} sup {hist[-1]:.2e} l2 {r.norm().item():.2e} cut {best[1]:.0e} t {best[2]} lam {p[0].item():.9f} mu {p[1].item():.9f}", flush=True)
        stall = stall + 1 if gain < 0.1 else 0
        if hist[-1] < tol or stall >= 2: break
    F.MAP = mapof(p[0]); rr = S.residual(p, G, shape); sup = torch.cat([rr[:-2], rr[-1:]]).abs().max().item(); del G; torch.cuda.empty_cache()
    return p, sup, len(hist) - 1, mapof(p[0])


def offgrid(q, own, yq):
    F.MAP = own; G2 = S.peak_grid(q[0].item(), yq, own, n0, h=h, R=RR * 1.5, device=DEV, shift=0.37, widen=1.2)
    rr = S.residual(q, G2, shape); off = torch.cat([rr[:-2], rr[-1:]]).abs().max().item(); del G2; torch.cuda.empty_cache(); return off


def standard(q, own):
    """Scale factor a with a Theta(y/a) ~ -(D/2)|y|^mu: y* in the usual normalisation is a * (y* here)."""
    F.MAP = own; al = 1 / (1 + q[0].item()); one = torch.ones(1, dtype=S.DT, device=DEV)
    amp = -2 * (al / 2) ** (q[1].item() / 2) * (F.basis3(one, n0)[0][0] @ q[2:2 + n0]).item()      # Theta ~ -(amp/2)|y|^mu
    return (amp / S.D) ** (1 / (q[1].item() - 1))


out = open("runs/sing_track.jsonl", "a"); t0 = time.time(); new = []


def record(q, own, yq, pk, fwy, sup, off, its, fn, base):
    lam, mu = q[0].item(), q[1].item(); u0 = (1 + lam) - lam / mu; c = q[2:2 + n0].abs(); a = standard(q, own); ystd = a * yq
    print(f"lam {lam:.9f} mu {mu:.9f} U'(0) {u0:.9f} sup {sup:.1e} OFF {off:.1e} its {its} peak {pk:.3f}@{ystd:.7f} (scale {a:.8f}) FWHM {fwy:.4e} pk2HW {pk*pk*fwy/2:.4f} eps {own[1]:.2e} c_mid {c[n0//2-2:n0//2+1].max().item():.0e} c_last {c[-3:].max().item():.0e} t={time.time()-t0:.0f}s", flush=True)
    out.write(json.dumps(dict(case="sing-track4-" + tag, lam=lam, mu=mu, up0=u0, shape=list(shape), quad_h=h, res_sup=sup, res_offgrid=off, iters=its, peak=pk, y_peak=ystd, y_peak_raw=yq, scale=a, fwhm=fwy, map=list(own))) + "\n"); out.flush()
    new.append(dict(lam=lam, p=q.clone().cpu(), map=own, ypk=yq, njJ=(nj, J), peak=pk, fwhm=fwy, off=off, scale=a, ystd=ystd, center=S.CENTER)); torch.save(base + new, fn)


if mode == "polish":
    pmin = float(A[4]); every = int(A[5]) if len(A) > 5 else 1; cnt = 0
    for x in sols[-int(os.environ.get('TAIL', len(sols))):]:
        S.CENTER = None; p, own, ypk, pk, fwy = align(padto(x["p"].to(DEV)), tuple(x["map"]))
        if pk < pmin: continue
        cnt += 1
        if (cnt - 1) % every: continue
        print(f"  polish lam {p[0].item():.9f} mu {p[1].item():.9f} peak {pk:.3f}", flush=True)
        q, sup, its, mq = solve(p, own, ypk, pk, fwy)
        own2 = mq; pk2, yq, _, _, fwy = measure(q, mq); yq = S.CENTER; off = offgrid(q, own2, yq)    # no remap: the state stays in its solve map
        record(q, own2, yq, pk2, fwy, sup, off, its, f"runs/sing_track4_{tag}.pt", [])
else:
    ratio, nsteps = float(A[4]), int(A[5]); states = []; lr0 = math.log(ratio); lr = lr0; done = 0
    for x in sols[-NP:]:
        p, own, ypk, pk, fwy = align(padto(x["p"].to(DEV)), tuple(x["map"])); states.append((p, own, ypk, pk, fwy))
    while done < nsteps:
        use = states[-NP:]; ss = [math.log(st[3]) for st in use]; s_t = ss[-1] + lr; mp = use[-1][1]
        wts = [math.prod((s_t - ss[k]) / (ss[i] - ss[k]) for k in range(len(use)) if k != i) for i in range(len(use))]
        guess = sum(w_ * st[0] for w_, st in zip(wts, use))
        cur = (sum(w_ * st[1][0] for w_, st in zip(wts, use)), math.exp(sum(w_ * math.log(st[1][1]) for w_, st in zip(wts, use))))
        yg = math.exp(sum(w_ * math.log(st[2]) for w_, st in zip(wts, use))); fg = math.exp(sum(w_ * math.log(st[4]) for w_, st in zip(wts, use)))
        al_ = 1 / (1 + guess[0].item()); sg_ = (1 + yg * yg) ** (-al_ / 2)
        cur = (cur[0], EPSF * fg * 2 * al_ * sg_ * yg / (1 + yg * yg))        # map width = predicted FWHM in x = 2s-1
        print(f"  try peak {math.exp(s_t):.3f} predicted lam {guess[0].item():.9f} mu {guess[1].item():.9f} eps {cur[1]:.2e} t={time.time()-t0:.0f}s", flush=True)
        q, sup, its, mq = solve(guess, cur, yg, math.exp(s_t), fg); ok = sup < 1e-5; off = float("nan")
        if ok:
            own = mq; pk, yq, _, _, fwy = measure(q, mq); yq = S.CENTER; off = offgrid(q, own, yq); ok = off < ACC
        if not ok:
            lr /= 2; print(f"  rejected (sup {sup:.1e}, off {off:.1e}); log-step -> {lr:.4f}", flush=True)
            if lr < 1e-3: print("stopping: step too small"); break
            continue
        record(q, own, yq, pk, fwy, sup, off, its, f"runs/sing_track4_{tag}.pt", sols)
        states.append((q, own, yq, pk, fwy)); states = states[-NP:]; done += 1; lr = min(lr * 1.5, lr0)
