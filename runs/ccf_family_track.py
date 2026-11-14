"""Track the lam-family of CCF profiles downward from the first unstable profile.
At each fixed lam: solve the coefficients with rows = collocation for y >= ymin + normalisation Omega'(0) = -d,
WITHOUT the origin relation. The basis forces Theta ~ y^2 at 0, so away from an admissible lam the equation
cannot hold near the origin; those rows are dropped. Reported: g(lam) = lam + 2 - 2 U'(0), which vanishes
exactly at a smooth profile (origin relation). The map (x0, eps) follows the measured peak.
Usage: ccf_family_track.py src.pt sx0 seps n h ymin lam_start lam_end step tag"""
import sys, os, json, math, time, torch
from gpu_solver import ccf as C, ccf_fast as F
a = sys.argv
src, sx0, seps, n, h, ymin = a[1], float(a[2]), float(a[3]), int(a[4]), float(a[5]), float(a[6])
l0, l1, step, tag = float(a[7]), float(a[8]), float(a[9]), a[10]
DEV = os.environ.get("DEV", "cpu"); d = 0.1
G = C.Grid(h=h, L=40.0, Lc=22.0, device=DEV, n_gl=int(os.environ.get("NGL", 128)))
p = torch.load(src).to(DEV).reshape(-1); cur_map = (sx0, seps)
out = open("runs/ccf_family_track.jsonl", "a"); sols = []
yy = torch.linspace(1.5, 6.0, 90001, dtype=C.DT, device=DEV)


def measure(p, mp):
    F.MAP = mp
    lam = p[0]; al = 1 / (1 + lam.item())
    Om = F._basis(lam, yy, p.numel() - 1)[1] @ p[1:]
    i = Om.abs().argmax(); pk = Om[i].abs().item()
    hw = (Om.abs() > pk / 2).nonzero().flatten()
    sf = lambda v: (1 + v * v) ** (-al / 2)
    return pk, yy[i].item(), 2 * sf(yy[i].item()) - 1, 2 * (sf(yy[hw[0]].item()) - sf(yy[hw[-1]].item()))


def remap(p, old, new, n):
    j = torch.arange(n, dtype=C.DT, device=DEV); t = torch.cos(math.pi * (j + 0.5) / n)
    x0, eps = new; a1, a2 = math.asinh((1 - x0) / eps), math.asinh((1 + x0) / eps)
    s = ((x0 + eps * torch.sinh((a1 + a2) * (t + 1) / 2 - a2) + 1) / 2).clamp(1e-300, 1.0)
    F.MAP = old; M = F.basis3(s, p.numel() - 1)[0] @ p[1:]
    F.MAP = new; return torch.cat([p[:1], torch.linalg.solve(F.basis3(s, n)[0], M)])


lam = l0; t0 = time.time()
while lam >= l1 - 1e-12:
    pk, ypk, x0, fw = measure(p, cur_map)
    new_map = (x0, max(0.33 * fw, 0.004))
    p = remap(p, cur_map, new_map, n); cur_map = new_map; p[0] = lam
    F.MODE.update(norm="origin", origin_row=False, d=d, ymin=ymin)
    p, hist = F.lm(p, G, lam_free=False, max_iter=120)
    r = F.residual(p[0], p[1:], G); ops = F.operators(p[0], G, n); u0 = (ops[4] @ p[1:]).item()
    g = lam + 2 - 2 * u0
    pk, ypk, _, fw = measure(p, cur_map); cc = p[1:].abs()
    print(f"lam {lam:.6f} g {g:+.3e} sup {r.abs().max().item():.1e} its {len(hist)} peak {pk:.4f}@{ypk:.4f} fwhm_x {fw:.4f} map ({cur_map[0]:.4f},{cur_map[1]:.4f}) "
          f"c_mid {cc[n//2-2:n//2+1].max().item():.0e} c_last {cc[-3:].max().item():.0e} t={time.time()-t0:.0f}s", flush=True)
    out.write(json.dumps(dict(case="2a-ccf-family-track-" + tag, n_basis=n, quad_h=h, ymin=ymin, lam=lam, g=g, res_sup=r.abs().max().item(), iters=len(hist),
                              peak=pk, y_peak=ypk, map=list(cur_map), Up0=u0)) + "\n"); out.flush()
    sols.append(dict(lam=lam, p=p.clone().cpu(), map=cur_map, g=g))
    torch.save(sols, f"runs/ccf_family_track_{tag}.pt")
    lam = round(lam - step, 10)
