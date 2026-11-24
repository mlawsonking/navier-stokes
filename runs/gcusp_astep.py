"""Continuation in the advection parameter a away from a = -1, starting from stored a = -1 states (CPU).
Equation (exact for every a; see gcusp_dmuda.py): integrated form
    R_I = -lam Theta + ((1+lam) y - U) Theta' + (a+1) Gint,   Gint = U'(0)(y Omega - Theta) + int_0^y (U - U'(0) y) Omega',
origin row  -lam + (1+lam-U'(0)) mu + (a+1) U'(0)(mu-1) = 0, extra far-field column Theta_C = (1-s)^(mu/2).
Gint is linear in the fields given the state, so its Jacobian is formed columnwise (4th-order differences in tau and
cumulative trapezoid with end correction on the uniform-in-tau rows). Predictor: tangent d x/da from the same Jacobian.
Modes: mu   : mu fixed (= MU, default 2: the smooth profile), lam free, amplitude row   [validation against lam_1(a)]
       peak : |Omega(y_g)| fixed, peak at y_g, lam and mu free (sing_track4 rows)
Grid and map are kept from the a = -1 state (fine for moderate |a+1| at fixed peak; check the printed off-grid value).
Usage: gcusp_astep.py SRC.pt idx mode a_end n_steps     env THREADS, H, ROWS, MU"""
import sys, os, math, torch
from gpu_solver import ccf_sing as S, ccf_fast as F
torch.set_num_threads(int(os.environ.get("THREADS", 8)))
DEV = "cpu"; DT = S.DT
src, k, mode, a_end, nst = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
h = float(os.environ.get("H", 0.0125)); RR = float(os.environ.get("ROWS", 6))
x = torch.load(src, map_location="cpu")[k]; nj, J = x["njJ"]; n0 = x["p"].numel() - 2 - nj * J; shape = (n0, nj, J); p = x["p"].to(DT)


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


lam0 = p[0].item(); yg = x["ypk"]; MU = float(os.environ.get("MU", 2.0))
if mode == "peak":
    if "peak" in x: S.CENTER = yg; eps = x["map"][1]
    else:
        S.CENTER = None; F.MAP = tuple(x["map"]); eps = x["map"][1]; S.CENTER = yg
        p = S.remap0(p, tuple(x["map"]), (2 * (1 + yg * yg) ** (-0.5 / (1 + lam0)) - 1, eps), shape, DEV)
    mapof = lambda lam: (2 * (1 + yg * yg) ** (-0.5 / (1 + float(lam))) - 1, eps)
else:
    S.CENTER = None; eps = x["map"][1]; mapof = lambda lam: tuple(x["map"]); p[1] = MU
F.MAP = mapof(lam0)
G = S.peak_grid(lam0, yg, mapof(lam0), n0, h=h, R=RR, device=DEV); nrow = G.y.numel()
G2 = S.peak_grid(lam0, yg, mapof(lam0), n0, h=h * 0.8, R=RR * 1.3, device=DEV, shift=0.37, widen=1.4)
al0 = 1 / (1 + lam0); sg = (1 + yg * yg) ** (-al0 / 2); dl = 0.025 * eps / (2 * al0 * sg * yg / (1 + yg * yg))
ygt = torch.tensor([yg, yg + dl, yg - dl], dtype=DT)
F.MAP = mapof(lam0); target = -(S._fields(p[0], p[1], ygt, shape)[1] @ p[2:])[0].item()


def fields(pf, Gr):
    lam, mu, c, C = pf[0], pf[1], pf[2:-1], pf[-1]
    F.MAP = mapof(lam)
    BTh, BOm, AU, up0, wt, nvec = S.operators(lam, mu, Gr, shape)
    ThCs, OmCs = extra(lam, mu, Gr.ys); ThC, OmC = extra(lam, mu, Gr.y)
    UC = (2 * Gr.y / math.pi) * (Gr.K @ ThCs - 1 / Gr.Y); up0C = -(2 / math.pi) * (Gr.w @ (OmCs / Gr.ys))
    BTh = torch.cat([BTh, ThC[:, None]], 1); BOm = torch.cat([BOm, OmC[:, None]], 1); AU = torch.cat([AU, UC[:, None]], 1)
    up0 = torch.cat([up0, up0C.reshape(1)]); nvec = torch.cat([nvec, torch.tensor([(1 / (1 + lam.item()) / 2) ** (mu.item() / 2)], dtype=DT)])
    Bg = torch.cat([S._fields(lam, mu, ygt, shape)[1], extra(lam, mu, ygt)[1][:, None]], 1)
    return BTh, BOm, AU, up0, wt, nvec, Bg


def resid(pf, a, Gr=None, want=False):
    Gr = Gr or G; hh = h if Gr is G else h * 0.8
    lam, mu, cc = pf[0], pf[1], pf[2:]
    BTh, BOm, AU, up0, wt, nvec, Bg = Fd = fields(pf, Gr)
    Th, Om, U, u0 = BTh @ cc, BOm @ cc, AU @ cc, up0 @ cc
    Omt = deriv(Om, hh); Gi = u0 * (Gr.y * Om - Th) + cumint((U - u0 * Gr.y) * Omt, hh)
    V = (1 + lam) * Gr.y - U
    R = (-lam * Th + V * Om + (a + 1) * Gi) / wt
    org = (-lam + (1 + lam - u0) * mu + (a + 1) * u0 * (mu - 1)).reshape(1)
    if mode == "mu":
        r = torch.cat([R, (nvec @ cc + S.D / 2).reshape(1), org]); ga = torch.cat([Gi / wt, torch.zeros(1, dtype=DT), (u0 * (mu - 1)).reshape(1)])
    else:
        o = (Bg @ cc) / target; r = torch.cat([R, org, (o[0] + 1).reshape(1), (100 * (o[1] - o[2])).reshape(1)])
        ga = torch.cat([Gi / wt, (u0 * (mu - 1)).reshape(1), torch.zeros(2, dtype=DT)])
    return (r, ga, Fd, Th, Om, U, u0, Omt, V) if want else r


def jac(pf, a):
    r, ga, Fd, Th, Om, U, u0, Omt, V = resid(pf, a, want=True)
    BTh, BOm, AU, up0, wt, nvec, Bg = Fd; lam, mu = pf[0], pf[1]
    y = G.y[:, None]
    dGi = (G.y * Om - Th)[:, None] * up0[None, :] + u0 * (y * BOm - BTh) \
        + cumint((AU - y * up0[None, :]) * Omt[:, None] + (U - u0 * G.y)[:, None] * deriv(BOm, h), h)
    Jc = (-lam * BTh + V[:, None] * BOm - Om[:, None] * AU + (a + 1) * dGi) / wt[:, None]
    oc = ((-mu + (a + 1) * (mu - 1)) * up0)[None]
    if mode == "mu":
        Jc = torch.cat([Jc, nvec[None], oc]); free = (0,)
    else:
        Bf = Bg / target; Jc = torch.cat([Jc, oc, Bf[0:1], 100 * (Bf[1:2] - Bf[2:3])]); free = (0, 1)
    cols = []
    for kk in free:
        e = torch.zeros_like(pf); e[kk] = 2e-7
        cols.append(((8 * (resid(pf + e, a) - resid(pf - e, a)) - (resid(pf + 2 * e, a) - resid(pf - 2 * e, a))) / (12 * 2e-7))[:, None])
    return r, ga, torch.cat(cols + [Jc], 1), free


def unpack(pf, dx, free):
    q = pf.clone(); q[list(free)] += dx[:len(free)]; q[2:] += dx[len(free):]; return q


def tsolve(Jm, rhs):
    d = Jm.norm(dim=0).clamp_min(1e-300); U_, sv, Vh = torch.linalg.svd(Jm / d, full_matrices=False); co = U_.T @ rhs
    return [(cut, (Vh[sv > cut * sv[0]].T @ (co[sv > cut * sv[0]] / sv[sv > cut * sv[0]])) / d) for cut in (1e-6, 1e-8, 1e-10, 1e-12)]


def newton(pf, a, its=12):
    for it in range(its):
        r, ga, Jm, free = jac(pf, a); rn = r.norm().item(); best = None
        for cut, dx in tsolve(Jm, -r):
            t = 1.0
            while t > 1e-3:
                q = unpack(pf, t * dx, free); nn = resid(q, a).norm().item()
                if math.isfinite(nn) and nn < rn: break
                t /= 2
            if t > 1e-3 and (best is None or nn < best[0]): best = (nn, q, cut, t)
        if best is None: break
        pf = best[1]; print(f"     newton {it} |r| {rn:.2e} -> {best[0]:.2e} (cut {best[2]:.0e}, t {best[3]})", flush=True)
        if best[0] > 0.5 * rn or best[0] < 1e-12: break
    return pf


def report(pf, a):
    r = resid(pf, a); r2 = resid(pf, a, Gr=G2)
    coll, off = r[:nrow].abs().max().item(), r2[:G2.y.numel()].abs().max().item()
    print(f"a {a:+.5f} lam {pf[0].item():.9f} mu {pf[1].item():.9f} C {pf[-1].item():+.5f}  coll {coll:.1e} OFF {off:.1e} extra rows {r[nrow:].abs().max().item():.1e}", flush=True)


pf = torch.cat([p, torch.zeros(1, dtype=DT)]); a = -1.0
print(f"state {src}[{k}] mode {mode} n0 {n0} rows {nrow} target peak {target:.3f} at y {yg:.5f}", flush=True)
pf = newton(pf, a, its=6); report(pf, a)
da = (a_end - a) / nst; prev = None
for st in range(nst):
    r, ga, Jm, free = jac(pf, a); dx = tsolve(Jm, -ga)[2][1]
    print(f"   tangent at a {a:+.4f}: dlam/da {(dx[0].item() if mode == 'mu' else dx[0].item()):+.6f}" + (f" dmu/da {dx[1].item():+.6f}" if mode == "peak" else ""), flush=True)
    a = a + da; pf = newton(unpack(pf, da * dx, free), a); report(pf, a)
