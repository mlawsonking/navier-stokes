"""Continue the FIRST UNSTABLE branch in the gCLM parameter a (smooth profiles, lam free, origin row).
Looking for how the branch ends (fold back into the second unstable branch?).
Usage: gclm_u1_branch.py src.pt sx0 seps n h a_start a_end step tag"""
import sys, os, json, math, time, torch
from gpu_solver import ccf as C, ccf_fast as F
A = sys.argv
src, sx0, seps, n, h, a0, a1, step, tag = A[1], float(A[2]), float(A[3]), int(A[4]), float(A[5]), float(A[6]), float(A[7]), float(A[8]), A[9]
DEV = os.environ.get("DEV", "cpu"); d = 0.1
G = C.Grid(h=h, L=40.0, Lc=22.0, device=DEV, n_gl=int(os.environ.get("NGL", 128)))
L = torch.load(src); cur = (sx0, seps)
pad = lambda v: torch.cat([v.to(DEV).reshape(-1), torch.zeros(max(0, n + 1 - v.numel()), dtype=C.DT, device=DEV)])
p = pad(L[-1]["p"] if isinstance(L, list) else L)
seed = (pad(L[-2]["p"]), tuple(L[-2]["map"]), L[-2]["a"], L[-1]["a"]) if isinstance(L, list) and len(L) > 1 else None
torch.set_num_threads(int(os.environ.get("THREADS", 6)))
CL = float(os.environ.get('CLUSTER', 0.0))
yy = torch.linspace(0.5, 12.0, 115001, dtype=C.DT, device=DEV)
F.MODE.update(norm="origin", origin_row=True, d=d, ymin=0.0)


def measure(p, mp):
    F.MAP = mp; lam = p[0]; al = 1 / (1 + lam.item())
    Om = F._basis(lam, yy, p.numel() - 1)[1] @ p[1:]
    i = Om.abs().argmax(); pk = Om[i].abs().item(); hw = (Om.abs() > pk / 2).nonzero().flatten()
    sf = lambda v: (1 + v * v) ** (-al / 2)
    return pk, yy[i].item(), 2 * sf(yy[i].item()) - 1, 2 * (sf(yy[hw[0]].item()) - sf(yy[hw[-1]].item()))


def remap(p, old, new, n):
    j = torch.arange(n, dtype=C.DT, device=DEV); t = torch.cos(math.pi * (j + 0.5) / n)
    x0, eps = new; b1, b2 = math.asinh((1 - x0) / eps), math.asinh((1 + x0) / eps)
    s = ((x0 + eps * torch.sinh((b1 + b2) * (t + 1) / 2 - b2) + 1) / 2).clamp(1e-300, 1.0)
    F.MAP = old; M = F.basis3(s, p.numel() - 1)[0] @ p[1:]
    F.MAP = new; return torch.cat([p[:1], torch.linalg.solve(F.basis3(s, n)[0], M)])


out = open("runs/gclm_u1_branch.jsonl", "a"); sols = []; prevmap = None; a = a0; sgn = 1.0 if a1 > a0 else -1.0; t0 = time.time(); prev = None; ap = aprev = None
if seed: prev, prevmap, aprev, ap = seed
while (a - a1) * sgn <= 1e-12:
    pk, ypk, x0, fw = measure(p, cur)
    new = (x0, max(float(os.environ.get("EPSF", 0.33)) * fw, 0.004)); p = remap(p, cur, new, n); cur = new
    guess = p.clone()
    if CL: G = C.Grid(h=h, L=40.0, Lc=22.0, device=DEV, n_gl=128, cluster=(math.asinh(ypk), 0.5, CL * 0.5))
    if prev is not None:                       # secant predictor in a (previous solution carried to the same map)
        # M ~ const/beta as lam -> 0 (beta = lam/(2(1+lam))), so extrapolate beta*c, not c
        pr = remap(prev, prevmap, cur, n); fz = (a - ap) / (ap - aprev); be = lambda l: l / (2 * (1 + l))
        lg = p[0] + fz * (p[0] - pr[0]); wg = be(p[0]) * p[1:] + fz * (be(p[0]) * p[1:] - be(pr[0]) * pr[1:])
        guess = torch.cat([lg.reshape(1), wg / be(lg)])
    q, hist = F.gn(guess, G, a=a, lam_free=True, max_iter=30, rc=float(os.environ.get('RC', 1e-9)), verbose=bool(os.environ.get('VERBOSE')))
    r = F.residual(q[0], q[1:], G, a); sup = r.abs().max().item()
    G2 = C.Grid(h=h * 0.8, L=40.0, Lc=22.0, device=DEV, n_gl=128, cluster=(math.asinh(ypk) + 0.011, 0.4, 0.38))
    off = F.residual(q[0], q[1:], G2, a).abs().max().item(); del G2      # residual on a grid the fit never saw
    pk, ypk, _, fw = measure(q, cur); cc = q[1:].abs()
    print(f"a {a:+.4f} lam {q[0].item():.12f} sup {sup:.1e} OFF {off:.1e} its {len(hist)} peak {pk:.4f}@{ypk:.3f} fwhm_x {fw:.4f} c_mid {cc[n//2-2:n//2+1].max().item():.0e} c_last {cc[-3:].max().item():.0e} t={time.time()-t0:.0f}s", flush=True)
    out.write(json.dumps(dict(case="gclm-u1-branch-" + tag, a=a, n_basis=n, quad_h=h, lam=q[0].item(), res_sup=sup, res_offgrid=off, iters=len(hist), peak=pk, y_peak=ypk, map=list(cur))) + "\n"); out.flush()
    if off > 1e-6:
        print("stopping: not converged"); break
    if ap is None or abs(a - ap) > 1e-12: prev, prevmap, aprev = p, cur, ap
    p, ap = q, a
    if aprev is None: prev = None
    sols.append(dict(a=a, p=q.clone().cpu(), map=cur)); torch.save(sols, f"runs/gclm_u1_branch_{tag}.pt")
    a = round(a + sgn * step, 10)
