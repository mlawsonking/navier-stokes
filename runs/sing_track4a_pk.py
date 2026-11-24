"""Checks of the general-a mu = 2 crossings found by runs/sing_track4a.py (same equation, rows and grids).
  pk     SRC.pt idx tag ratio n_steps     family at FIXED a (that of state idx) marched in the peak height: target *= ratio
                                          per step (ratio < 1 goes down); Lagrange predictor in ln(peak) (first step: a
                                          third of the step, previous state as predictor). Records go to runs/sing_track4a.jsonl
                                          with case sing-track4a-<tag>.
  smooth SRC.pt idx tag                   SMOOTH PIN: the state idx re-solved with the smooth-only representation:
                                          no non-smooth origin blocks (J = 0), mu = 2 fixed, lam free, peak height FREE
                                          (rows: collocation, origin relation, peak at y_g); prints lam, peak, on/off.
  osc    SRC.pt idx                       sign changes of Omega and number of local extrema of |Omega| on y > 0.
env as sing_track4a (DEV, THREADS, ACCEPT, CAP, EPSF, ROWS, NPRED, REMAP)."""
import sys, os, math, json, time, torch
import importlib.util
spec = importlib.util.spec_from_file_location("t4a", os.path.join(os.path.dirname(__file__), "sing_track4a.py"))
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
S, F, DT, DEV = T.S, T.F, T.DT, T.DEV
A = sys.argv; mode = A[1]


def load(src, idx):
    st = torch.load(src, map_location="cpu")
    return st, st[idx]


def as_t4a(st):
    """a stored sing_track2/x1 state (a = -1, its own map, CENTER None) -> the sing_track4a format (map centred on y_g)."""
    if "shape" in st: return st
    nj, J = st["njJ"]; p = st["p"].to(DT); n0 = p.numel() - 2 - nj * J; yg = st["ypk"]; eps = st["map"][1]; lam = p[0].item()
    S.CENTER = None; F.MAP = tuple(st["map"]); S.CENTER = yg
    p = S.remap0(p, tuple(st["map"]), (2 * (1 + yg * yg) ** (-0.5 / (1 + lam)) - 1, eps), (n0, nj, J), DEV)
    al = 1 / (1 + lam); sg = (1 + yg * yg) ** (-al / 2); fwy = eps / (2 * al * sg * yg / (1 + yg * yg))
    F.MAP = (2 * (1 + yg * yg) ** (-0.5 / (1 + lam)) - 1, eps)
    target = (S._fields(p[0], p[1], torch.tensor([yg], dtype=DT), (n0, nj, J))[1] @ p[2:]).abs().item()
    return dict(a=-1.0, pf=torch.cat([p, torch.zeros(1, dtype=DT)]), yg=yg, eps=eps, target=target, fwy=fwy, shape=(n0, nj, J))


def osc(pf, P, fwy):
    """sign changes of Omega and local extrema of |Omega| on y > 0 (log grid plus a fine grid around the peak)."""
    F.MAP = P.mapof(pf[0])
    y1 = torch.exp(torch.linspace(math.log(1e-4), math.log(1e8), 200001, dtype=DT, device=DEV))
    y2 = torch.linspace(P.yg - 200 * fwy, P.yg + 200 * fwy, 40001, dtype=DT, device=DEV)
    y = torch.sort(torch.cat([y1, y2]))[0]
    Om = T.om_at(pf, y, P.shape)
    sc = int((torch.sign(Om[1:]) != torch.sign(Om[:-1])).sum().item())
    d = Om.abs().diff(); ext = int(((d[1:] * d[:-1]) < 0).sum().item())
    i = Om.abs().argmax()
    return sc, ext, Om.abs().max().item(), y[i].item()


if mode == "pk":
    src, idx, tag, ratio, nst = A[2], int(A[3]), A[4], float(A[5]), int(A[6])
    store_in, st = load(src, idx); shape, yg, a = st["shape"], st["yg"], st["a"]
    store = [st]; fn = f"runs/sing_track4a_{tag}.pt"; done = 0; lr0 = math.log(ratio); lr = lr0 / 3; fails = 0
    print(f"family march at a {a:+.5f} from peak {st['target']:.3f}, ratio {ratio}", flush=True)
    while done < nst:
        use = store[-T.NP:]; ls = [math.log(s["target"]) for s in use]; lt = ls[-1] + lr; tgt = math.exp(lt)
        w = T.lagrange(ls, lt) if len(use) > 1 else [1.0]
        eps_t = math.exp(sum(wi * math.log(s["eps"]) for wi, s in zip(w, use))); fwy_t = math.exp(sum(wi * math.log(s["fwy"]) for wi, s in zip(w, use)))
        P = T.Problem(shape, yg, eps_t, tgt, fwy_t, use[-1]["pf"][0].item())
        mapped = []
        for s in use:
            q = s["pf"].to(DEV); Pm = (2 * (1 + yg * yg) ** (-0.5 / (1 + q[0].item())) - 1, s["eps"])
            mapped.append(T.remap_pf(q, Pm, P.mapof(q[0]), shape))
        guess = sum(wi * m for wi, m in zip(w, mapped))
        print(f"  try peak {tgt:.3f} predicted lam {guess[0].item():.9f} mu {guess[1].item():.9f} eps {eps_t:.3e}", flush=True)
        pf, sup, Jm, ga = T.solve(P, guess, a); off = T.offgrid(P, pf, a)
        if not T.accept_ok(sup, off):
            lr /= 2; fails += 1; print(f"  rejected (sup {sup:.1e}, off {off:.1e}); log-step -> {lr:.4f}", flush=True)
            if fails > 6: print("stopping"); break
            continue
        pk, ypk, fwy, fwx = T.measure(pf, P, fwy_t)
        eps_new = T.EPSF * fwx
        if abs(math.log(eps_new / P.eps)) > float(os.environ.get("REMAP", 0.02)):
            q = T.remap_pf(pf, P.mapof(pf[0]), (P.mapof(pf[0])[0], eps_new), shape); P2 = T.Problem(shape, yg, eps_new, tgt, fwy, pf[0].item())
            print(f"  map width {P.eps:.3e} -> {eps_new:.3e}: re-solve", flush=True)
            q, sup2, _, _ = T.solve(P2, q, a); off2 = T.offgrid(P2, q, a)
            if T.accept_ok(sup2, off2): pf, P, sup, off = q, P2, sup2, off2; pk, ypk, fwy, fwx = T.measure(pf, P, fwy)
        T.record(tag, a, pf, P, sup, off, pk, ypk, fwy, store, fn); done += 1; fails = 0
        lr = math.copysign(min(abs(lr) * 1.5, abs(lr0)), lr0)

elif mode == "smooth":
    # smooth SRC.pt idx tag [a_end da]: smooth pin at the state's a, then (optionally) continue the SMOOTH profile in a
    src, idx, tag = A[2], int(A[3]), A[4]; a_end = float(A[5]) if len(A) > 6 else None; da = float(A[6]) if len(A) > 6 else 0.0
    _, st = load(src, idx); st = as_t4a(st); sh0 = st["shape"]; n0 = sh0[0]; yg, a = st["yg"], st["a"]
    n0s = int(os.environ.get("N0S", n0)); shape = (n0s, 1, 0)          # N0S: resolution check with a bigger smooth basis
    q0 = st["pf"].to(DEV)
    pf = torch.cat([q0[:2], torch.zeros(n0s, dtype=DT, device=DEV), q0[-1:]]); pf[1] = 2.0
    P = T.Problem(shape, yg, st["eps"], st["target"], st["fwy"], pf[0].item())
    print(f"smooth pin at a {a:+.5f}: start lam {pf[0].item():.9f} (state mu {q0[1].item():.9f}); origin blocks max |c| {q0[2 + n0:-1].abs().max().item():.1e}", flush=True)
    # project Theta of the state (mu_old, with origin blocks) onto the smooth basis by weighted LSQ on the rows:
    # the block coefficients are large and nearly cancel against N_0 near mu = 2, so simply dropping them is O(1) off
    Pfull = T.Problem(sh0, yg, st["eps"], st["target"], st["fwy"], pf[0].item())
    BTh_o = Pfull.fields(q0, P.G)[0]; Th_old = BTh_o @ q0[2:]
    BTh_n, _, _, _, wtn, _, _ = P.fields(pf, P.G)
    cfit = torch.linalg.lstsq(BTh_n / wtn[:, None], (Th_old / wtn)[:, None]).solution[:, 0]; pf[2:] = cfit
    print(f"  projected onto the smooth basis: max weighted misfit {((BTh_n @ cfit - Th_old) / wtn).abs().max().item():.1e}", flush=True)
    n = P.nrow(); keep_rows = lambda r: torch.cat([r[:n + 1], r[n + 2:]])     # drop the peak-height row: height free

    def step(p, dx):
        q = p.clone(); q[0] += dx[0]; q[2:] += dx[1:]; return q

    def smooth_newton(pf, a):
        resid = lambda p: keep_rows(P.resid(p, a))
        def jac(p):
            r, ga, Jm = P.jac(p, a); Jm = torch.cat([Jm[:, :1], Jm[:, 2:]], 1)    # drop the mu column (mu = 2 fixed)
            return keep_rows(r), keep_rows(Jm)
        r, Jm = jac(pf); t0 = time.time()
        for it in range(20):
            best = None
            for cut, dx in T.tsvd_steps(Jm, -r):
                t = min(1.0, T.CAP / max(abs(dx[0].item()), 1e-300)); nn = resid(step(pf, t * dx)).norm().item()
                if math.isfinite(nn) and (best is None or nn < best[0]): best = (nn, cut, t, dx)
            if best is None or best[0] >= r.norm().item():
                # no cutoff gains with the full (capped) step: damp the most conservative one
                best = None
                for cut, dx in T.tsvd_steps(Jm, -r)[:3]:
                    t = 0.5
                    while t > 1e-3:
                        nn = resid(step(pf, t * dx)).norm().item()
                        if nn < r.norm().item(): break
                        t /= 2
                    if t > 1e-3 and (best is None or nn < best[0]): best = (nn, cut, t, dx)
                if best is None: break
            gain = 1 - best[0] / r.norm().item(); pf = step(pf, best[2] * best[3]); r, Jm = jac(pf)
            sup = r[:n + 1].abs().max().item()
            print(f"    gn {it} sup {sup:.2e} l2 {r.norm().item():.2e} cut {best[1]:.0e} t {best[2]:.3g} lam {pf[0].item():.10f} C {pf[-1].item():+.6f}  t={time.time() - t0:.0f}s", flush=True)
            if sup < 2e-9 or (gain < 0.02 and it > 3): break
        return pf, r[:n + 1].abs().max().item()

    hist = []; fwy = st["fwy"]
    avals = [a] + ([a + da * k for k in range(1, int(round((a_end - a) / da)) + 1)] if a_end is not None else [])
    for ak in avals:
        guess = pf if len(hist) < 2 else hist[-1][1] + (ak - hist[-1][0]) / (hist[-1][0] - hist[-2][0]) * (hist[-1][1] - hist[-2][1])
        pf, sup = smooth_newton(guess, ak); off = T.offgrid(P, pf, ak)
        pk, ypk, fwy, fwx = T.measure(pf, P, fwy); sc, ext, pk2, ypk2 = osc(pf, P, fwy); hist.append((ak, pf.clone()))
        eps_new = T.EPSF * fwx
        if abs(math.log(eps_new / P.eps)) > 0.05:          # keep the map width at the FWHM: remap state and history
            old = lambda q: (P.mapof(q[0])[0], P.eps)
            Pn = T.Problem(shape, yg, eps_new, pk, fwy, pf[0].item())
            hist = [(x, T.remap_pf(q, old(q), Pn.mapof(q[0]), shape)) for x, q in hist]; pf = hist[-1][1].clone()
            P = Pn; n = P.nrow(); print(f"  map width -> {eps_new:.3e} (rows {n})", flush=True)
        print(f"SMOOTH a {ak:+.5f} lam {pf[0].item():.10f} mu 2 (fixed) U'(0) {(pf[0].item() + 2) / (1 - ak):.8f} sup {sup:.1e} OFF {off:.1e} peak {pk:.4f} at {ypk:.7f} (y_g {yg:.7f}) "
              f"FWHM {fwy:.4e}  Omega sign changes {sc}, extrema of |Omega| {ext}", flush=True)
        with open("runs/sing_track4a.jsonl", "a") as fo:
            fo.write(json.dumps(dict(case="sing-track4a-smooth-" + tag, a=ak, lam=pf[0].item(), mu=2.0, C=pf[-1].item(), res_sup=sup, res_offgrid=off, peak=pk,
                                     y_g=yg, fwhm=fwy, shape=list(shape), sign_changes=sc, extrema=ext, src=src, idx=idx)) + chr(10))
        torch.save(dict(a=ak, pf=pf.cpu(), yg=yg, eps=P.eps, target=pk, fwy=fwy, shape=shape, off=off, sup=sup), f"runs/sing_track4a_smooth_{tag}.pt")

elif mode == "smootha":
    # smootha SRC.pt idx tag : SMOOTH PIN at FIXED PEAK HEIGHT (that of state idx) with lam AND a free (trap 14: freeing the
    # peak height at fixed a means a several-percent move along a sharp family, which Gauss-Newton does not make).
    # Representation smooth-only (J = 0, mu = 2), rows: collocation, origin, Omega(y_g) = -peak, peak at y_g.
    # Unknowns [lam, a, c, C]; the a column is the exact dR/da (Gint/wt, U'(0)(mu-1)).
    src, idx, tag = A[2], int(A[3]), A[4]
    _, st = load(src, idx); st = as_t4a(st); sh0 = st["shape"]; n0 = sh0[0]; yg, a = st["yg"], st["a"]
    # N0S: size of the smooth basis (the dropped origin blocks also carried resolution: N_0 alone at n0 = 1024 stalls at 8e-6)
    n0s = int(os.environ.get("N0S", n0)); shape = (n0s, 1, 0)
    q0 = st["pf"].to(DEV); pf = torch.cat([q0[:2], torch.zeros(n0s, dtype=DT, device=DEV), q0[-1:]]); pf[1] = 2.0
    P = T.Problem(shape, yg, st["eps"], st["target"], st["fwy"], pf[0].item())
    Pfull = T.Problem(sh0, yg, st["eps"], st["target"], st["fwy"], pf[0].item())
    Th_old = Pfull.fields(q0, P.G)[0] @ q0[2:]; BTh_n, _, _, _, wtn, _, _ = P.fields(pf, P.G)
    pf[2:] = torch.linalg.lstsq(BTh_n / wtn[:, None], (Th_old / wtn)[:, None]).solution[:, 0]
    print(f"smooth pin at fixed peak {st['target']:.4f}: start a {a:+.6f} lam {pf[0].item():.9f} (state mu {q0[1].item():.9f}); projection misfit "
          f"{((BTh_n @ pf[2:] - Th_old) / wtn).abs().max().item():.1e}", flush=True)
    n = P.nrow(); t0 = time.time()

    def jac(p, a):
        r, ga, Jm = P.jac(p, a); return r, torch.cat([Jm[:, :1], ga[:, None], Jm[:, 2:]], 1)

    def step(p, a, dx):
        q = p.clone(); q[0] += dx[0]; q[2:] += dx[2:]; return q, a + dx[1].item()

    r, Jm = jac(pf, a)
    for it in range(20):
        best = None
        for cut, dx in T.tsvd_steps(Jm, -r):
            t = min(1.0, T.CAP / max(abs(dx[0].item()), abs(dx[1].item()), 1e-300)); q, aq = step(pf, a, t * dx); nn = P.resid(q, aq).norm().item()
            if math.isfinite(nn) and (best is None or nn < best[0]): best = (nn, cut, t, dx)
        if best is None or best[0] >= r.norm().item(): break
        gain = 1 - best[0] / r.norm().item(); pf, a = step(pf, a, best[2] * best[3]); r, Jm = jac(pf, a)
        sup = torch.cat([r[:n], r[n:n + 1]]).abs().max().item()
        print(f"    gn {it} sup {sup:.2e} l2 {r.norm().item():.2e} cut {best[1]:.0e} t {best[2]:.3g} a {a:+.8f} lam {pf[0].item():.10f} C {pf[-1].item():+.6f}  t={time.time() - t0:.0f}s", flush=True)
        if sup < float(os.environ.get("TOL", 2e-9)) or (gain < 0.05 and it > 2): break
    sup = torch.cat([r[:n], r[n:n + 1]]).abs().max().item(); off = T.offgrid(P, pf, a)
    pk, ypk, fwy, fwx = T.measure(pf, P, st["fwy"]); sc, ext, _, _ = osc(pf, P, fwy)
    print(f"SMOOTH(fixed peak) a {a:+.8f} lam {pf[0].item():.10f} mu 2 sup {sup:.1e} OFF {off:.1e} peak {pk:.4f} (pinned {st['target']:.4f}) at {ypk:.7f} "
          f"FWHM {fwy:.4e}  Omega sign changes {sc}, extrema of |Omega| {ext}", flush=True)
    with open("runs/sing_track4a.jsonl", "a") as fo:
        fo.write(json.dumps(dict(case="sing-track4a-smootha-" + tag, a=a, lam=pf[0].item(), mu=2.0, C=pf[-1].item(), res_sup=sup, res_offgrid=off, peak=pk,
                                 y_g=yg, fwhm=fwy, shape=list(shape), sign_changes=sc, extrema=ext, src=src, idx=idx)) + chr(10))
    torch.save(dict(a=a, pf=pf.cpu(), yg=yg, eps=P.eps, target=pk, fwy=fwy, shape=shape, off=off, sup=sup), f"runs/sing_track4a_smootha_{tag}.pt")

elif mode == "fold":
    # fold SRC.pt idx tag ratio n_steps : the SMOOTH branch (mu = 2 pinned) followed in the PEAK HEIGHT with lam AND a free.
    # Representation: the family's (non-smooth-capable) basis with mu fixed at 2, i.e. the blocks stay as extra
    # functions (their non-smooth content must come out ~0: printed as the smooth-basis misfit after each step).
    # Rows: collocation, origin, Omega(y_g) = -peak, peak at y_g. Unknowns [lam, a, c, C]; a-column = exact dR/da.
    # Peak is the parameter because a turns at the fold. First step: the state itself (mu reset to 2) at its own peak.
    src, idx, tag, ratio, nst = A[2], int(A[3]), A[4], float(A[5]), int(A[6])
    _, st0 = load(src, idx); st0 = as_t4a(st0); shape, yg = st0["shape"], st0["yg"]; n0 = shape[0]
    q = st0["pf"].to(DEV).clone()                      # mu is moved to 2 by the tangent predictor below
    if os.environ.get("SB") and shape[2] > 0 and abs(q[1].item() - 2) < 1e-6:
        # SB=1: continue in the SMOOTH-ONLY basis (J = 0): project the (mu = 2) state onto it first
        sh_s = (n0, 1, 0); Pf = T.Problem(shape, yg, st0["eps"], st0["target"], st0["fwy"], q[0].item())
        Ps = T.Problem(sh_s, yg, st0["eps"], st0["target"], st0["fwy"], q[0].item())
        Th = Pf.fields(q, Ps.G)[0] @ q[2:]; qs = torch.cat([q[:2], torch.zeros(n0 + 1, dtype=DT, device=DEV)]); qs[1] = 2.0
        B, _, _, _, wt, _, _ = Ps.fields(qs, Ps.G); qs[2:] = torch.linalg.lstsq(B / wt[:, None], (Th / wt)[:, None]).solution[:, 0]
        print(f"  smooth-only basis: projection misfit {((B @ qs[2:] - Th) / wt).abs().max().item():.1e}", flush=True)
        q = qs; shape = sh_s; st0 = dict(st0, shape=sh_s)
    store = [dict(st0, pf=q.cpu(), a=st0["a"])]; fn = f"runs/sing_track4a_{tag}.pt"
    lr0 = math.log(ratio); first = True; done = 0; fails = 0; lr = lr0
    if os.environ.get("HIST"):     # resume a fold march: take the last NPRED states of SRC (a fold store) as history
        store = [dict(x) for x in torch.load(src, map_location="cpu")[-T.NP:]]; shape = store[-1]["shape"]; n0 = shape[0]; first = False

    def fsolve(P, pf, a, max_iter=15):
        n = P.nrow()
        def jac(p, a):
            r, ga, Jm = P.jac(p, a); return r, torch.cat([Jm[:, :1], ga[:, None], Jm[:, 2:]], 1)
        def step(p, a, dx):
            q = p.clone(); q[0] += dx[0]; q[2:] += dx[2:]; return q, a + dx[1].item()
        r, Jm = jac(pf, a); stall = 0
        for it in range(max_iter):
            best = None
            # each cutoff with damped variants: the full step of the right direction (it moves a by ~1e-3 at fixed peak,
            # i.e. the width by ~1 percent) can be worse than a no-op refit of the coefficients, which then wins and stalls
            for cut, dx in T.tsvd_steps(Jm, -r):
                t0 = min(1.0, T.CAP / max(abs(dx[0].item()), abs(dx[1].item()), 1e-300))
                for t in (t0, t0 / 2, t0 / 4):
                    qq, aq = step(pf, a, t * dx); nn = P.resid(qq, aq).norm().item()
                    if math.isfinite(nn) and (best is None or nn < best[0]): best = (nn, cut, t, dx)
            if best is None or best[0] >= r.norm().item():
                best = None
                for cut, dx in T.tsvd_steps(Jm, -r)[:3]:
                    t = 0.5
                    while t > 1e-3:
                        qq, aq = step(pf, a, t * dx); nn = P.resid(qq, aq).norm().item()
                        if nn < r.norm().item(): break
                        t /= 2
                    if t > 1e-3 and (best is None or nn < best[0]): best = (nn, cut, t, dx)
                if best is None: break
            gain = 1 - best[0] / r.norm().item(); pf, a = step(pf, a, best[2] * best[3]); r, Jm = jac(pf, a)
            sup = torch.cat([r[:n], r[n:n + 1]]).abs().max().item()
            print(f"    gn {it} sup {sup:.2e} l2 {r.norm().item():.2e} cut {best[1]:.0e} t {best[2]:.3g} a {a:+.9f} lam {pf[0].item():.10f}", flush=True)
            stall = stall + 1 if gain < 0.1 else 0
            if sup < float(os.environ.get("TOL", 2e-9)) or stall >= 2: break
        return pf, a, torch.cat([r[:n], r[n:n + 1]]).abs().max().item()

    def smooth_misfit(P, pf):
        Ps = T.Problem((n0, 1, 0), yg, P.eps, P.target, P.fwy if hasattr(P, "fwy") else 1e-3, pf[0].item())
        Th = P.fields(pf, P.G)[0] @ pf[2:]; ps = torch.cat([pf[:2], torch.zeros(n0 + 1, dtype=DT, device=DEV)])
        B, _, _, _, wt, _, _ = Ps.fields(ps, P.G); c = torch.linalg.lstsq(B / wt[:, None], (Th / wt)[:, None]).solution[:, 0]
        return ((B @ c - Th) / wt).abs().max().item()

    while done < nst:
        use = store[-T.NP:]
        if first:
            tgt = use[-1]["target"]; w = [1.0]
        else:
            ls = [math.log(s["target"]) for s in use]; lt = ls[-1] + lr; tgt = math.exp(lt); w = T.lagrange(ls, lt) if len(use) > 1 else [1.0]
        eps_t = math.exp(sum(wi * math.log(s["eps"]) for wi, s in zip(w, use))); fwy_t = math.exp(sum(wi * math.log(s["fwy"]) for wi, s in zip(w, use)))
        a_g = sum(wi * s["a"] for wi, s in zip(w, use))
        if len(use) == 1 and not first:
            # single state: scale the map width with the inner law FWHM ~ peak^-(1-a) (a fixed map width at a 4 percent
            # higher peak left a representation floor of 8e-7 that Newton cannot remove)
            sc = (tgt / use[-1]["target"]) ** (-(1 - use[-1]["a"])); eps_t *= sc; fwy_t *= sc
        P = T.Problem(shape, yg, eps_t, tgt, fwy_t, use[-1]["pf"][0].item())
        mapped = []
        for s in use:
            qs = s["pf"].to(DEV); Pm = (2 * (1 + yg * yg) ** (-0.5 / (1 + qs[0].item())) - 1, s["eps"])
            mapped.append(T.remap_pf(qs, Pm, P.mapof(qs[0]), shape))
        guess = sum(wi * m for wi, m in zip(w, mapped))
        if first and abs(guess[1].item() - 2.0) > 1e-9:
            # do NOT just overwrite mu: that changes the basis functions under a sharp peak (residual 0.2) and the Newton
            # correction is far outside its linear range (it stalls on coefficient-only refits). Move along the family
            # tangent at fixed peak instead: d(state)/da with mu free, by the a-step that brings mu to 2.
            r_, ga_, Jm_ = P.jac(guess, a_g); dx = T.tsvd_steps(Jm_, -ga_)[3][1]
            dA = (2.0 - guess[1].item()) / dx[1].item(); guess = guess + dA * dx; a_g = a_g + dA
            print(f"  tangent predictor: dmu/da {dx[1].item():+.5f}, a-step {dA:+.3e}, mu -> {guess[1].item():.9f}", flush=True)
        if (not first) and len(use) == 1:
            # one smooth state only: tangent in ln(peak) with a free (d r/d ln target = +1 on the peak-height row)
            Pc = T.Problem(shape, yg, use[-1]["eps"], use[-1]["target"], use[-1]["fwy"], use[-1]["pf"][0].item())
            q1 = use[-1]["pf"].to(DEV); r_, ga_, Jm_ = Pc.jac(q1, use[-1]["a"]); J2 = torch.cat([Jm_[:, :1], ga_[:, None], Jm_[:, 2:]], 1)
            e = torch.zeros_like(r_); e[Pc.nrow() + 1] = 1.0; dx = T.tsvd_steps(J2, -e)[3][1]
            q2 = q1.clone(); q2[0] += lr * dx[0]; q2[2:] += lr * dx[2:]; a_g = use[-1]["a"] + lr * dx[1].item()
            Pm = (2 * (1 + yg * yg) ** (-0.5 / (1 + q2[0].item())) - 1, use[-1]["eps"])   # tangent is relative to the moving map
            guess = T.remap_pf(q2, Pm, P.mapof(q2[0]), shape)
            print(f"  peak tangent: da/dlnP {dx[1].item():+.4e} dlam/dlnP {dx[0].item():+.4e}", flush=True)
        guess[1] = 2.0
        print(f"  try peak {tgt:.4f} predicted a {a_g:+.8f} lam {guess[0].item():.10f} eps {eps_t:.3e}", flush=True)
        pf, a, sup = fsolve(P, guess, a_g); off = T.offgrid(P, pf, a)
        if not T.accept_ok(sup, off):
            lr /= 2; fails += 1; print(f"  rejected (sup {sup:.1e}, off {off:.1e}); log-step -> {lr:.4f}", flush=True)
            if fails > 5 or first: print("stopping"); break
            continue
        pk, ypk, fwy, fwx = T.measure(pf, P, fwy_t)
        mis = smooth_misfit(P, pf)
        print(f"FOLD a {a:+.9f} lam {pf[0].item():.10f} mu 2 sup {sup:.1e} OFF {off:.1e} peak {pk:.4f} (target {tgt:.4f}) smooth-basis misfit {mis:.1e} "
              f"block max|c| {(pf[2 + n0:-1].abs().max().item() if pf.numel() > n0 + 3 else 0.0):.1e}", flush=True)
        with open("runs/sing_track4a.jsonl", "a") as fo:
            fo.write(json.dumps(dict(case="sing-track4a-fold-" + tag, a=a, lam=pf[0].item(), mu=2.0, C=pf[-1].item(), res_sup=sup, res_offgrid=off,
                                     peak=pk, target=tgt, y_g=yg, fwhm=fwy, smooth_misfit=mis, shape=list(shape))) + chr(10))
        eps_new = T.EPSF * fwx
        if abs(math.log(eps_new / P.eps)) > float(os.environ.get("REMAP", 0.05)):
            pf = T.remap_pf(pf, P.mapof(pf[0]), (P.mapof(pf[0])[0], eps_new), shape); P.eps = eps_new
        store.append(dict(a=a, pf=pf.clone().cpu(), yg=yg, eps=P.eps, target=tgt, fwy=fwy, shape=shape, off=off, sup=sup)); torch.save(store, fn)
        if first: store = store[-1:]; first = False
        else: done += 1; fails = 0; lr = math.copysign(min(abs(lr) * 1.5, abs(lr0)), lr0)

elif mode == "osc":
    src, idx = A[2], int(A[3]); _, st = load(src, idx); st = as_t4a(st); P = T.Problem(st["shape"], st["yg"], st["eps"], st["target"], st["fwy"], st["pf"][0].item())
    sc, ext, pk, ypk = osc(st["pf"].to(DEV), P, st["fwy"])
    print(f"{src}[{idx}] a {st['a']:+.5f} lam {st['pf'][0].item():.9f} mu {st['pf'][1].item():.9f} peak {pk:.4f} at {ypk:.6f}: Omega sign changes {sc}, extrema of |Omega| {ext}")
else:
    raise SystemExit(__doc__)
