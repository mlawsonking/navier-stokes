"""General-a non-smooth family at FIXED peak height: march in the advection parameter a (sing_track4 + gcusp_astep).
Equation (exact for every a; derivation in runs/gcusp_dmuda.py):
    R_I = -lam Theta + ((1+lam) y - U) Theta' + (a+1) Gint = 0,  Gint = U'(0)(y Omega - Theta) + int_0^y (U - U'(0) y) Omega',
    origin row  -lam + (1+lam-U'(0)) mu + (a+1) U'(0)(mu-1) = 0,
    extra far-field column Theta_C = (1-s)^(mu/2) with coefficient C (lam C = (a+1) Gint(inf); C = 0 at a = -1).
Rows/unknowns as sing_track4: collocation, origin, Omega(y_g) = -target, Omega(y_g+dl) = Omega(y_g-dl) (the peak IS at
y_g: fixes the scale, so y_g is held FIXED for the whole march); unknowns [lam, mu, c, C]; map centre follows lam
(mapof), map width = EPSF x measured FWHM, remapped after every accepted step; accurate kernel (S.CENTER = y_g);
CAP on the lam/mu move per Newton iteration; ladder of SVD cutoffs, best actual residual wins.
Gint: 4th-order differences in tau on the (uniform in tau) rows + trapezoid with end correction; its Jacobian columnwise.
Acceptance: off-grid residual (denser shifted grid) below ACCEPT and close to the fit residual.
Usage:
  sing_track4a.py march SRC.pt i0,i1[,i2] TARGET tag a_end da       start at a = -1 from the stored states i0.. (Lagrange
       in ln peak to the TARGET peak height), solve there (acceptance vs the a = -1 data), then march a in steps da.
  sing_track4a.py resume TRACK4A.pt tag a_end da                     continue a march (last NPRED states)
env: DEV (cpu|cuda, default cpu), THREADS, ACCEPT (default 3e-7), CAP (default 1e-4), EPSF (1.0), ROWS (6), NPRED (3),
     REMAP (re-fit the map width to the measured FWHM and re-solve when it drifted by more than this in log, default 0.02)"""
import sys, os, math, json, time, torch
from gpu_solver import ccf_sing as S, ccf_fast as F
DEV = os.environ.get("DEV", "cpu"); torch.set_num_threads(int(os.environ.get("THREADS", 8))); DT = S.DT
ACC = float(os.environ.get("ACCEPT", 3e-7)); CAP = float(os.environ.get("CAP", 1e-4)); EPSF = float(os.environ.get("EPSF", 1.0))
RR = float(os.environ.get("ROWS", 6)); NP = int(os.environ.get("NPRED", 3)); h = 0.0125
A = sys.argv; mode = A[1] if (__name__ == "__main__" and len(A) > 1) else "import"   # importable (sing_track4a_pk.py)


# ---------------- quadrature helpers (as gcusp_astep) ----------------
def deriv(f, hh):
    d = torch.empty_like(f)
    d[2:-2] = (8 * (f[3:-1] - f[1:-3]) - (f[4:] - f[:-4])) / (12 * hh)
    d[0] = (-25 * f[0] + 48 * f[1] - 36 * f[2] + 16 * f[3] - 3 * f[4]) / (12 * hh)
    d[1] = (-3 * f[0] - 10 * f[1] + 18 * f[2] - 6 * f[3] + f[4]) / (12 * hh)
    d[-1] = (25 * f[-1] - 48 * f[-2] + 36 * f[-3] - 16 * f[-4] + 3 * f[-5]) / (12 * hh)
    d[-2] = (3 * f[-1] + 10 * f[-2] - 18 * f[-3] + 6 * f[-4] - f[-5]) / (12 * hh)
    return d


def cumint(f, hh):
    T = torch.cat([torch.zeros_like(f[:1]), torch.cumsum((f[1:] + f[:-1]) / 2 * hh, 0)])
    fp = deriv(f, hh)
    return T - hh * hh / 12 * (fp - fp[:1])


def extra(lam, mu, y):
    al = 1 / (1 + lam); l = torch.log1p(y * y); s = torch.exp(-al / 2 * l); oms = -torch.expm1(-al / 2 * l)
    return oms ** (mu / 2), (mu / 2) * oms ** (mu / 2 - 1) * al * y * s / (1 + y * y)


class Problem:
    """Residual/Jacobian at fixed (a, y_g, target, map width eps) on the grid built for the current predictor."""
    def __init__(self, shape, yg, eps, target, fwy, lam_g):
        self.shape, self.yg, self.eps, self.target = shape, yg, eps, target
        S.CENTER = yg
        dl = 0.05 * fwy / 2
        self.ygt = torch.tensor([yg, yg + dl, yg - dl], dtype=DT, device=DEV)
        F.MAP = self.mapof(lam_g)
        self.G = S.peak_grid(lam_g, yg, self.mapof(lam_g), shape[0], h=h, R=RR, device=DEV)

    def mapof(self, lam):
        return (2 * (1 + self.yg * self.yg) ** (-0.5 / (1 + float(lam))) - 1, self.eps)

    def fields(self, pf, Gr):
        lam, mu = pf[0], pf[1]; F.MAP = self.mapof(lam)
        BTh, BOm, AU, up0, wt, nvec = S.operators(lam, mu, Gr, self.shape)
        ThCs, OmCs = extra(lam, mu, Gr.ys); ThC, OmC = extra(lam, mu, Gr.y)
        UC = (2 * Gr.y / math.pi) * (Gr.K @ ThCs - 1 / Gr.Y); up0C = -(2 / math.pi) * (Gr.w @ (OmCs / Gr.ys))
        BTh = torch.cat([BTh, ThC[:, None]], 1); BOm = torch.cat([BOm, OmC[:, None]], 1); AU = torch.cat([AU, UC[:, None]], 1)
        up0 = torch.cat([up0, up0C.reshape(1)])
        Bg = torch.cat([S._fields(lam, mu, self.ygt, self.shape)[1], extra(lam, mu, self.ygt)[1][:, None]], 1)
        # U' = H Omega on the rows (Omega odd: kernel 2t/(pi (y^2 - t^2))), the same accurate half-step rule as for U
        BOs = torch.cat([S._fields(lam, mu, Gr.ys, self.shape)[1], OmCs[:, None]], 1)
        AUp = (2 / math.pi) * (Gr.K @ (Gr.ys[:, None] * BOs))
        return BTh, BOm, AU, up0, wt, Bg, AUp

    def resid(self, pf, a, Gr=None, want=False):
        Gr = Gr or self.G; lam, mu, cc = pf[0], pf[1], pf[2:]
        BTh, BOm, AU, up0, wt, Bg, AUp = Fd = self.fields(pf, Gr)
        Th, Om, U, u0, Up = BTh @ cc, BOm @ cc, AU @ cc, up0 @ cc, AUp @ cc
        # Gint = U Omega - int_0^y U' Omega dy: no derivative of the sharp Omega (4th-order differences of Omega had a
        # 1e-5 relative error at the peak, seen as an off-grid residual of 1e-7 per 0.01 in a). Prefix 0..y_0 from
        # U' - U'(0) ~ |y|^(mu-1): u0 Theta_0 + (U'_0 - u0) Omega_0 y_0/(2 mu - 1). Rows are uniform in tau: dy = y_tau dtau.
        yt = deriv(Gr.y, h); m2 = 2 * mu - 1
        pre = u0 * Th[0] + (Up[0] - u0) * Om[0] * Gr.y[0] / m2
        Gi = U * Om - pre - cumint(Up * Om * yt, h)
        Omt = yt
        V = (1 + lam) * Gr.y - U
        R = (-lam * Th + V * Om + (a + 1) * Gi) / wt
        org = (-lam + (1 + lam - u0) * mu + (a + 1) * u0 * (mu - 1)).reshape(1)
        o = (Bg @ cc) / self.target
        r = torch.cat([R, org, (o[0] + 1).reshape(1), (100 * (o[1] - o[2])).reshape(1)])
        if not want: return r
        ga = torch.cat([Gi / wt, (u0 * (mu - 1)).reshape(1), torch.zeros(2, dtype=DT, device=DEV)])
        return r, ga, Fd, Th, Om, U, u0, Omt, V

    def jac(self, pf, a):
        r, ga, Fd, Th, Om, U, u0, Omt, V = self.resid(pf, a, want=True)
        BTh, BOm, AU, up0, wt, Bg, AUp = Fd; lam, mu = pf[0], pf[1]; G = self.G; yt = Omt; m2 = 2 * mu - 1
        Up = AUp @ pf[2:]
        dpre = up0 * Th[0] + u0 * BTh[0] + ((AUp[0] - up0) * Om[0] + (Up[0] - u0) * BOm[0]) * G.y[0] / m2
        dGi = AU * Om[:, None] + U[:, None] * BOm - dpre[None, :] - cumint((AUp * Om[:, None] + Up[:, None] * BOm) * yt[:, None], h)
        Jc = (-lam * BTh + V[:, None] * BOm - Om[:, None] * AU + (a + 1) * dGi) / wt[:, None]
        oc = ((-mu + (a + 1) * (mu - 1)) * up0)[None]; Bf = Bg / self.target
        Jc = torch.cat([Jc, oc, Bf[0:1], 100 * (Bf[1:2] - Bf[2:3])])
        cols = []
        for kk in (0, 1):
            e = torch.zeros_like(pf); e[kk] = 2e-7
            cols.append(((8 * (self.resid(pf + e, a) - self.resid(pf - e, a)) - (self.resid(pf + 2 * e, a) - self.resid(pf - 2 * e, a))) / (12 * 2e-7))[:, None])
        return r, ga, torch.cat(cols + [Jc], 1)

    def nrow(self, Gr=None):
        return (Gr or self.G).y.numel()


def om_at(pf, y, shape):
    out = []
    for yy in y.split(8192):
        out.append(S._fields(pf[0], pf[1], yy, shape)[1] @ pf[2:-1] + pf[-1] * extra(pf[0], pf[1], yy)[1])
    return torch.cat(out)


def measure(pf, P, fwy_guess):
    """peak value and position, FWHM in y and in x = 2s-1 (internal normalisation), on local fine grids around y_g."""
    F.MAP = P.mapof(pf[0]); al = 1 / (1 + pf[0].item()); sf = lambda v: (1 + v * v) ** (-al / 2)
    y0 = P.yg; w = 3 * fwy_guess
    for _ in range(6):
        yl = torch.linspace(y0 - w, y0 + w, 20001, dtype=DT, device=DEV); Om = om_at(pf, yl, P.shape).abs(); i = Om.argmax()
        y0, pk = yl[i].item(), Om[i].item(); hw = (Om > pk / 2).nonzero().flatten()
        if hw[0] > 0 and hw[-1] < 20000 and hw.numel() > 400:
            i0, i1 = hw[0].item(), hw[-1].item()
            a_ = yl[i0 - 1].item() + (yl[i0] - yl[i0 - 1]).item() * (pk / 2 - Om[i0 - 1].item()) / (Om[i0] - Om[i0 - 1]).item()
            b_ = yl[i1].item() + (yl[i1 + 1] - yl[i1]).item() * (Om[i1].item() - pk / 2) / (Om[i1] - Om[i1 + 1]).item()
            return pk, y0, b_ - a_, 2 * (sf(a_) - sf(b_))
        w = w * 3 if not (hw[0] > 0 and hw[-1] < 20000) else w / 2
    raise RuntimeError("FWHM measurement failed")


def tsvd_steps(Jm, rhs):
    d = Jm.norm(dim=0).clamp_min(1e-300); U_, sv, Vh = torch.linalg.svd((Jm / d).cpu(), full_matrices=False); co = U_.T @ rhs.cpu()
    out = []
    for cut in (1e-4, 1e-6, 1e-8, 1e-9, 1e-10, 1e-12):
        k = sv > cut * sv[0]; out.append((cut, (Vh[k].T @ (co[k] / sv[k])).to(DEV) / d))
    return out


def solve(P, pf, a, max_iter=15, tol=float(os.environ.get("TOL", 2e-9)), verbose=True):
    r, ga, Jm = P.jac(pf, a); n = P.nrow(); hist = [r.norm().item()]; stall = 0
    for it in range(max_iter):
        if not (torch.isfinite(Jm).all() and torch.isfinite(r).all()): break
        best = None
        for cut, st in tsvd_steps(Jm, -r):
            t = min(1.0, CAP / max(abs(st[0].item()), abs(st[1].item()), 1e-300))
            nn = P.resid(pf + t * st, a).norm().item()
            if math.isfinite(nn) and (best is None or nn < best[0]): best = (nn, cut, t, st)
        if best is None or best[0] >= r.norm().item():
            st = tsvd_steps(Jm, -r)[0][1]; t = 0.5; best = None
            while t > 1e-3:
                nn = P.resid(pf + t * st, a).norm().item()
                if nn < r.norm().item(): best = (nn, 1e-4, t, st); break
                t /= 2
            if best is None: break
        gain = 1 - best[0] / r.norm().item(); pf = pf + best[2] * best[3]; r, ga, Jm = P.jac(pf, a)
        sup = torch.cat([r[:n], r[n:n + 1]]).abs().max().item(); hist.append(sup)
        if verbose: print(f"    gn {it} sup {sup:.2e} l2 {r.norm().item():.2e} cut {best[1]:.0e} t {best[2]:.3g} lam {pf[0].item():.9f} mu {pf[1].item():.9f} C {pf[-1].item():+.5f}", flush=True)
        stall = stall + 1 if gain < 0.1 else 0
        if sup < tol or stall >= 2: break
    sup = torch.cat([r[:n], r[n:n + 1]]).abs().max().item()
    return pf, sup, Jm, ga


def offgrid(P, pf, a):
    F.MAP = P.mapof(pf[0]); G2 = S.peak_grid(pf[0].item(), P.yg, P.mapof(pf[0]), P.shape[0], h=h, R=RR * 1.5, device=DEV, shift=0.37, widen=1.2)
    rr = P.resid(pf, a, Gr=G2); n2 = G2.y.numel(); off = torch.cat([rr[:n2], rr[n2:n2 + 1]]).abs().max().item()
    del G2
    if DEV == "cuda": torch.cuda.empty_cache()
    return off


def standard(pf, P):
    """scale factor b with b Theta(y/b) ~ -(D/2)|y|^mu (usual normalisation); y* there = b * y_g. Includes the C column."""
    F.MAP = P.mapof(pf[0]); al = 1 / (1 + pf[0].item()); one = torch.ones(1, dtype=DT, device=DEV); n0 = P.shape[0]
    amp = -2 * (al / 2) ** (pf[1].item() / 2) * ((F.basis3(one, n0)[0][0] @ pf[2:2 + n0]).item() + pf[-1].item())
    return (amp / S.D) ** (1 / (pf[1].item() - 1))


out = open("runs/sing_track4a.jsonl", "a"); t0 = time.time()


def record(tag, a, pf, P, sup, off, pk, ypk, fwy, store, fn):
    lam, mu = pf[0].item(), pf[1].item(); b = standard(pf, P)
    print(f"ACCEPT a {a:+.5f} lam {lam:.9f} mu {mu:.9f} C {pf[-1].item():+.6f} sup {sup:.1e} OFF {off:.1e} peak {pk:.4f} (target {P.target:.4f}) "
          f"pos {ypk:.7f} (y_g {P.yg:.7f}) FWHM {fwy:.4e} eps {P.eps:.3e} y*_std {b * P.yg:.6f} t={time.time() - t0:.0f}s", flush=True)
    out.write(json.dumps(dict(case="sing-track4a-" + tag, a=a, lam=lam, mu=mu, C=pf[-1].item(), res_sup=sup, res_offgrid=off, peak=pk,
                              target=P.target, y_g=P.yg, fwhm=fwy, eps=P.eps, ystd=b * P.yg, shape=list(P.shape))) + "\n"); out.flush()
    store.append(dict(a=a, pf=pf.clone().cpu(), yg=P.yg, eps=P.eps, target=P.target, fwy=fwy, shape=P.shape, off=off, sup=sup))
    torch.save(store, fn)


def remap_pf(pf, old, new, shape):
    q = S.remap0(pf[:-1], old, new, shape, DEV); return torch.cat([q, pf[-1:]])


def lagrange(xs, x):
    return [math.prod((x - xs[k]) / (xs[i] - xs[k]) for k in range(len(xs)) if k != i) for i in range(len(xs))]


def accept_ok(sup, off):
    return off < ACC and sup < ACC


if mode == "march":
    src, idxs, target, tag, a_end, da = A[2], [int(v) for v in A[3].split(",")], float(A[4]), A[5], float(A[6]), float(A[7])
    sols = torch.load(src, map_location="cpu"); xs = [sols[i] for i in idxs]
    nj, J = xs[0]["njJ"]; n0 = xs[0]["p"].numel() - 2 - nj * J; shape = (n0, nj, J)
    # predictor at a = -1 in ln peak through the stored states (each in its own map; coefficients relative to the map)
    lp = [math.log(x["peak"]) for x in xs]; w_ = lagrange(lp, math.log(target))
    yg = math.exp(sum(wi * math.log(x["ypk"]) for wi, x in zip(w_, xs))); fwy = math.exp(sum(wi * math.log(x["fwhm"]) for wi, x in zip(w_, xs)))
    eps = math.exp(sum(wi * math.log(x["map"][1]) for wi, x in zip(w_, xs)))
    p = sum(wi * x["p"].to(DT) for wi, x in zip(w_, xs)).to(DEV); pf = torch.cat([p, torch.zeros(1, dtype=DT, device=DEV)])
    # the stored maps are centred on their own y_g: carry the interpolated coefficients to the map centred on the new y_g
    x0_old = sum(wi * x["map"][0] for wi, x in zip(w_, xs))
    P = Problem(shape, yg, eps, target, fwy, pf[0].item()); pf = remap_pf(pf, (x0_old, eps), P.mapof(pf[0]), shape)
    store = []; fn = f"runs/sing_track4a_{tag}.pt"; a = -1.0
    print(f"start: target peak {target} y_g {yg:.7f} eps {eps:.3e} predicted lam {pf[0].item():.9f} mu {pf[1].item():.9f} rows {P.nrow()} n0 {n0} DEV {DEV}", flush=True)
    pf, sup, Jm, ga = solve(P, pf, a); off = offgrid(P, pf, a); pk, ypk, fwy, fwx = measure(pf, P, fwy)
    record(tag, a, pf, P, sup, off, pk, ypk, fwy, store, fn)
    first_tangent = (Jm, ga)
elif mode == "resume":
    src, tag, a_end, da = A[2], A[3], float(A[4]), float(A[5]); store = torch.load(src, map_location="cpu"); fn = f"runs/sing_track4a_{tag}.pt"
    st = store[-1]; shape = st["shape"]; yg, target = st["yg"], st["target"]; a = st["a"]; first_tangent = None
elif mode != "import":
    raise SystemExit(__doc__)

da0 = da if mode != "import" else 0.0; fails = 0
while mode != "import" and (a_end - a) * math.copysign(1, da) > 1e-12:
    a_t = a + da
    use = store[-NP:]
    eps_t = math.exp(sum(wi * math.log(s["eps"]) for wi, s in zip(lagrange([s["a"] for s in use], a_t), use))) if len(use) > 1 else use[-1]["eps"]
    fwy_t = math.exp(sum(wi * math.log(s["fwy"]) for wi, s in zip(lagrange([s["a"] for s in use], a_t), use))) if len(use) > 1 else use[-1]["fwy"]
    lam_guess = use[-1]["pf"][0].item()
    P = Problem(shape, yg, eps_t, target, fwy_t, lam_guess)
    # carry every stored state to the predicted map (centre follows its own lam), then Lagrange in a
    mapped = []
    for s in use:
        q = s["pf"].to(DEV); Pm = (2 * (1 + yg * yg) ** (-0.5 / (1 + q[0].item())) - 1, s["eps"])
        mapped.append(remap_pf(q, Pm, P.mapof(q[0]), shape))
    if len(use) == 1:
        # first step: tangent of the a = -1 solve (same map); remap afterwards
        if first_tangent is not None:          # tangent in the map of the stored state
            Jm, ga = first_tangent; dx = tsvd_steps(Jm, -ga)[3][1]
            # the lam column of the Jacobian includes the map motion (mapof(lam)), so the tangent is relative to the
            # MOVING map: add it in place, then change only the width (remapping the centre too would count it twice)
            q = use[-1]["pf"].to(DEV) + da * dx; Pm = (2 * (1 + yg * yg) ** (-0.5 / (1 + q[0].item())) - 1, use[-1]["eps"])
            guess = remap_pf(q, Pm, P.mapof(q[0]), shape)
        else:                                  # tangent in the predicted map
            r_, ga, Jm = P.jac(mapped[0], a); guess = mapped[0] + da * tsvd_steps(Jm, -ga)[3][1]
    else:
        guess = sum(wi * m for wi, m in zip(lagrange([s["a"] for s in use], a_t), mapped))
    print(f"  try a {a_t:+.5f} predicted lam {guess[0].item():.9f} mu {guess[1].item():.9f} eps {eps_t:.3e} rows {P.nrow()} t={time.time() - t0:.0f}s", flush=True)
    try:
        pf, sup, Jm, ga = solve(P, guess, a_t); off = offgrid(P, pf, a_t)
    except RuntimeError as e:
        print("  solve error", e); sup = off = float("inf")
    if not accept_ok(sup, off):
        da /= 2; fails += 1; print(f"  rejected (sup {sup:.1e}, off {off:.1e}); da -> {da}", flush=True)
        if abs(da) < 1e-4 or fails > 6: print("stopping: step too small"); break
        continue
    pk, ypk, fwy, fwx = measure(pf, P, fwy_t)
    # refit the map width to the measured FWHM: remap and re-solve once if it drifted by more than 10 percent
    eps_new = EPSF * fwx
    if abs(math.log(eps_new / P.eps)) > float(os.environ.get("REMAP", 0.02)):
        q = remap_pf(pf, P.mapof(pf[0]), (P.mapof(pf[0])[0], eps_new), shape); P2 = Problem(shape, yg, eps_new, target, fwy, pf[0].item())
        print(f"  map width {P.eps:.3e} -> {eps_new:.3e}: re-solve", flush=True)
        q, sup2, _, _ = solve(P2, q, a_t); off2 = offgrid(P2, q, a_t)
        if accept_ok(sup2, off2): pf, P, sup, off = q, P2, sup2, off2; pk, ypk, fwy, fwx = measure(pf, P, fwy)
    a = a_t; record(tag, a, pf, P, sup, off, pk, ypk, fwy, store, fn); fails = 0; da = min(abs(da) * 1.5, abs(da0)) * math.copysign(1, da0)
