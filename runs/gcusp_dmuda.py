"""First-order perturbation in the advection parameter about the a = -1 (CCF) non-smooth family, CPU only.

Exact identity (any a): with Theta' = Omega, U = H Theta, V_a = (1+lam) y + a U,
    Omega + V_a Omega' - Omega U' = d/dy[ -lam Theta + ((1+lam) y - U) Theta' ] + (a+1) U Omega',
so the general-a profile equation in integrated ("Burgers") form is
    R_I = -lam Theta + ((1+lam) y - U) Theta' + (a+1) Gint(y) = 0,   Gint(y) = int_0^y U Omega' dy'.
Origin (Theta ~ A|y|^mu): mu (1 + lam + a U'(0)) = lam + (1+a) U'(0)  [a = -1: mu = lam/(1+lam-U'(0))],
correction exponents unchanged (mu + j(mu-1) + 2k). Far field: Theta ~ q y^(1-alpha) + C with
lam C = (a+1) Gint(inf), a CONSTANT the a = -1 basis cannot carry (it is r^beta s^lam): one extra column
Theta_C = (1-s)^(mu/2) (|y|^mu at the origin, 1 at infinity) is added.
At a = -1 the Jacobian is the ccf_sing Burgers Jacobian; dR/da = [Gint/wt, U'(0)(mu-1), 0...]. Solve
J dx = -dR/da by truncated SVD -> d lam/da, d mu/da along the chosen constraint:
  mode lam : lam fixed, amplitude row (sing_track2 / x1 normalisation)      -> d mu/da |_lam
  mode peak: |Omega(y_g)| fixed and the peak at y_g (sing_track4 rows), lam and mu free -> d(lam, mu)/da |_peak
Gint on the collocation rows (uniform in tau): 4th-order differences of Omega in tau, trapezoid + end correction;
a second value from every other row is printed (Richardson check).
Usage: gcusp_dmuda.py SRC.pt mode i1 [i2 ...]   env THREADS, H (0.0125), ROWS (6)"""
import sys, os, math, json, torch
from gpu_solver import ccf_sing as S, ccf_fast as F
torch.set_num_threads(int(os.environ.get("THREADS", 8)))
DEV = "cpu"; DT = S.DT
src, mode = sys.argv[1], sys.argv[2]; idxs = [int(v) for v in sys.argv[3:]]
sols = torch.load(src, map_location="cpu"); h = float(os.environ.get("H", 0.0125)); RR = float(os.environ.get("ROWS", 6))


def deriv(f, hh):
    d = torch.empty_like(f)
    d[2:-2] = (8 * (f[3:-1] - f[1:-3]) - (f[4:] - f[:-4])) / (12 * hh)
    d[0] = (-25 * f[0] + 48 * f[1] - 36 * f[2] + 16 * f[3] - 3 * f[4]) / (12 * hh)
    d[1] = (-3 * f[0] - 10 * f[1] + 18 * f[2] - 6 * f[3] + f[4]) / (12 * hh)
    d[-1] = (25 * f[-1] - 48 * f[-2] + 36 * f[-3] - 16 * f[-4] + 3 * f[-5]) / (12 * hh)
    d[-2] = (3 * f[-1] + 10 * f[-2] - 18 * f[-3] + 6 * f[-4] - f[-5]) / (12 * hh)
    return d


def cumint(f, hh):
    T = torch.cat([torch.zeros(1, dtype=DT), torch.cumsum((f[1:] + f[:-1]) / 2 * hh, 0)])
    fp = deriv(f, hh)
    return T - hh * hh / 12 * (fp - fp[0])


def extra(lam, mu, y):
    al = 1 / (1 + lam); l = torch.log1p(y * y); s = torch.exp(-al / 2 * l); oms = -torch.expm1(-al / 2 * l)
    return oms ** (mu / 2), (mu / 2) * oms ** (mu / 2 - 1) * al * y * s / (1 + y * y)


for k in idxs:
    x = sols[k]; nj, J = x["njJ"]; n0 = x["p"].numel() - 2 - nj * J; shape = (n0, nj, J); p = x["p"].to(DT)
    lam0 = p[0].item(); mu0 = p[1].item()
    if mode == "peak":
        yg = x["ypk"]
        if "peak" in x:          # sing_track4 state: solved with CENTER = ypk and map = mapof(lam)
            S.CENTER = yg; eps = x["map"][1]
        else:                    # sing_track2 state: carry to a map centred on its peak
            S.CENTER = None; F.MAP = tuple(x["map"]); eps = x["map"][1]
            S.CENTER = yg; p = S.remap0(p, tuple(x["map"]), (2 * (1 + yg * yg) ** (-0.5 / (1 + lam0)) - 1, eps), shape, DEV)
        mapof = lambda lam: (2 * (1 + yg * yg) ** (-0.5 / (1 + float(lam))) - 1, eps)
    else:
        S.CENTER = None; yg = x["ypk"]; eps = x["map"][1]; mapof = lambda lam: tuple(x["map"])
    F.MAP = mapof(lam0)
    G = S.peak_grid(lam0, yg, mapof(lam0), n0, h=h, R=RR, device=DEV)
    al0 = 1 / (1 + lam0); sg = (1 + yg * yg) ** (-al0 / 2); dydx = 1 / (2 * al0 * sg * yg / (1 + yg * yg))
    dl = 0.025 * eps * dydx
    ygt = torch.tensor([yg, yg + dl, yg - dl], dtype=DT)
    F.MAP = mapof(lam0); target = -(S._fields(p[0], p[1], ygt, shape)[1] @ p[2:])[0].item()

    def parts(pf):
        lam, mu, c, C = pf[0], pf[1], pf[2:-1], pf[-1]
        F.MAP = mapof(lam)
        BTh, BOm, AU, up0, wt, nvec = S.operators(lam, mu, G, shape)
        ThCs, OmCs = extra(lam, mu, G.ys); ThC, OmC = extra(lam, mu, G.y)
        UC = (2 * G.y / math.pi) * (G.K @ ThCs - 1 / G.Y); up0C = -(2 / math.pi) * (G.w @ (OmCs / G.ys))
        Bg = S._fields(lam, mu, ygt, shape)[1]; OmCg = extra(lam, mu, ygt)[1]
        ampC = (1 / (1 + lam) / 2) ** (mu / 2)
        return BTh, BOm, AU, up0, wt, nvec, ThC, OmC, UC, up0C, Bg, OmCg, ampC

    def resid(pf, want=False):
        lam, mu, c, C = pf[0], pf[1], pf[2:-1], pf[-1]
        BTh, BOm, AU, up0, wt, nvec, ThC, OmC, UC, up0C, Bg, OmCg, ampC = P = parts(pf)
        Th = BTh @ c + C * ThC; Om = BOm @ c + C * OmC; U = AU @ c + C * UC; u0 = up0 @ c + C * up0C
        V = (1 + lam) * G.y - U
        R = (-lam * Th + V * Om) / wt; org = (-lam + (1 + lam - u0) * mu).reshape(1)
        if mode == "lam":
            r = torch.cat([R, (nvec @ c + C * ampC + S.D / 2).reshape(1), org])
        else:
            o = (Bg @ c + C * OmCg) / target; r = torch.cat([R, org, (o[0] + 1).reshape(1), (100 * (o[1] - o[2])).reshape(1)])
        return (r, P, Th, Om, U, u0, V) if want else r

    pf = torch.cat([p, torch.zeros(1, dtype=DT)])
    r, P, Th, Om, U, u0, V = resid(pf, want=True)
    BTh, BOm, AU, up0, wt, nvec, ThC, OmC, UC, up0C, Bg, OmCg, ampC = P
    lam, mu = pf[0], pf[1]
    Jc = torch.cat([(-lam * BTh + V[:, None] * BOm - Om[:, None] * AU) / wt[:, None], ((-lam * ThC + V * OmC - Om * UC) / wt)[:, None]], 1)
    oc = torch.cat([-mu * up0, (-mu * up0C).reshape(1)])
    if mode == "lam":
        Jc = torch.cat([Jc, torch.cat([nvec, ampC.reshape(1) if torch.is_tensor(ampC) else torch.tensor([ampC], dtype=DT)])[None], oc[None]])
    else:
        Bf = torch.cat([Bg, OmCg[:, None]], 1) / target
        Jc = torch.cat([Jc, oc[None], Bf[0:1], 100 * (Bf[1:2] - Bf[2:3])])
    cols = []
    for kk in ((0, 1) if mode == "peak" else (1,)):
        e = torch.zeros_like(pf); e[kk] = 2e-7
        cols.append(((8 * (resid(pf + e) - resid(pf - e)) - (resid(pf + 2 * e) - resid(pf - 2 * e))) / (12 * 2e-7))[:, None])
    Jm = torch.cat(cols + [Jc], 1)
    # d R / d a
    # Gint = U'(0) (y Omega - Theta) + int_0^y (U - U'(0) y) Omega': the first part is exact, the remainder's
    # integrand is smaller by y^min(2, mu-1) at the origin, so starting the quadrature at the first row costs nothing
    Omt = deriv(Om, h); U0 = u0.item(); base = U0 * (G.y * Om - Th); f = (U - U0 * G.y) * Omt
    Gi = base + cumint(f, h)
    Gi2 = base[::2] + cumint(f[::2], 2 * h)          # coarser (every other row): Richardson check
    errG = (Gi[::2] - Gi2).abs().max().item() / Gi.abs().max().item()
    nrow = G.y.numel(); ga = torch.zeros(r.numel(), dtype=DT); ga[:nrow] = Gi / wt
    ga[nrow + (1 if mode == "lam" else 0)] = u0 * (mu - 1)
    d = Jm.norm(dim=0).clamp_min(1e-300); Us, sv, Vh = torch.linalg.svd(Jm / d, full_matrices=False); co = Us.T @ -ga
    print(f"[{k}] lam {lam0:.9f} mu {mu0:.9f} U'(0) {u0.item():.6f} target peak {target:.3f} at y {yg:.5f}  n0 {n0} rows {nrow}"
          f"  |r|sup coll {r[:G.y.numel()].abs().max().item():.1e} all {r.abs().max().item():.1e}  Gint(inf) {Gi[-1].item():+.6f} (rel. 2h-change {errG:.1e})", flush=True)
    res = {}
    for cut in (1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13):
        kp = sv > cut * sv[0]; dx = (Vh[kp].T @ (co[kp] / sv[kp])) / d; resn = (Jm @ dx + ga).norm().item() / ga.norm().item()
        if mode == "peak":
            dlam, dmu = dx[0].item(), dx[1].item(); dC = dx[-1].item()
        else:
            dlam, dmu = 0.0, dx[0].item(); dC = dx[-1].item()
        res[cut] = (dlam, dmu)
        print(f"   cut {cut:.0e} rank {int(kp.sum())}/{sv.numel()}  dlam/da {dlam:+.7f}  dmu/da {dmu:+.7f}  dC/da {dC:+.5f} (far-field balance Gint(inf)/lam {Gi[-1].item() / lam0:+.5f})  rel.res {resn:.1e}", flush=True)
    with open("runs/gcusp_dmuda.jsonl", "a") as fo:
        fo.write(json.dumps(dict(src=src, idx=k, mode=mode, lam=lam0, mu=mu0, peak=target, n0=n0, h=h, rows=RR,
                                 res={f"{c:.0e}": v for c, v in res.items()}, Ginf=Gi[-1].item(), errG=errG)) + "\n")
    if os.environ.get("DIAG"):
        kp = sv > 1e-10 * sv[0]; dx = (Vh[kp].T @ (co[kp] / sv[kp])) / d; rr = (Jm @ dx + ga)
        print("   residual by y-decade (max |J dx + g| on rows; max |g|):")
        ly = torch.log10(G.y)
        for lo in range(-6, 10):
            m = (ly >= lo) & (ly < lo + 1)
            if m.any(): print(f"     y 1e{lo:+d}: {rr[:nrow][m].abs().max().item():.2e}   g {ga[:nrow][m].abs().max().item():.2e}")
        print("   extra rows:", rr[nrow:].tolist())
    if os.environ.get("DIAG"):
        e2 = (Gi[::2] - Gi2).abs(); ly2 = torch.log10(G.y[::2])
        for lo in range(-6, 10):
            m = (ly2 >= lo) & (ly2 < lo + 1)
            if m.any(): print(f"     Gint 2h-change y 1e{lo:+d}: {e2[m].max().item():.2e}  |Gint| {Gi[::2][m].abs().max().item():.2e}  |base| {base[::2][m].abs().max().item():.2e}")
