"""Pin a SMOOTH CCF profile (mu = 2, smooth basis only, lam unknown) from a nearby state.
Usage: sing_pin2.py in.pt out.pt n0 h     in.pt: track list (last entry used) or a dict written by this script.
env: EPSF, EPSMIN, ROWS, ROUNDS. Grid: peak_grid (rows follow the basis density); off-grid check on a second grid."""
import sys, os, json, math, time, torch
from gpu_solver import ccf_sing as S, ccf_fast as F
DEV = "cuda"; A = sys.argv; n0, h = int(A[3]), float(A[4])
EPSF = float(os.environ.get("EPSF", 1.0)); EPSMIN = float(os.environ.get("EPSMIN", 4e-5)); RR = float(os.environ.get("ROWS", 6))
x = torch.load(A[1]); x = x[-1] if isinstance(x, list) else x
p = x["p"].to(DEV); cur = tuple(x["map"]); nj, J = x["njJ"]; nold = p.numel() - 2 - nj * J; shape = (n0, 1, 0)
def nodes(mp):
    j = torch.arange(n0, dtype=S.DT, device=DEV); t = torch.cos(math.pi * (j + 0.5) / n0)
    x0, eps = mp; b1, b2 = math.asinh((1 - x0) / eps), math.asinh((1 + x0) / eps)
    return (x0 + eps * torch.sinh((b1 + b2) * (t + 1) / 2 - b2) + 1) / 2
s = nodes(cur); F.MAP = cur
E, _ = S.E2(s, 1 - s, p[1], (nold, nj, J))                       # full ansatz at the state's own mu ...
Ntot = (E @ p[2:]) / (1 - s) ** (p[1] / 2)                        # ... divided by (1-s)^(mu/2): the smooth-basis start
q = torch.cat([p[:1], torch.tensor([2.0], dtype=S.DT, device=DEV), torch.linalg.solve(F.basis3(s, n0)[0], Ntot)])
yy = torch.exp(torch.linspace(math.log(0.3), math.log(60.0), 480001, dtype=S.DT, device=DEV))
def measure(q, mp):
    F.MAP = mp; al = 1 / (1 + q[0].item()); Om = (S._fields(q[0], q[1], yy, shape)[1] @ q[2:]).abs()
    i = Om.argmax(); pk = Om[i].item(); hw = (Om > pk / 2).nonzero().flatten(); sf = lambda v: (1 + v * v) ** (-al / 2)
    return pk, yy[i].item(), 2 * sf(yy[i].item()) - 1, 2 * (sf(yy[hw[0]].item()) - sf(yy[hw[-1]].item()))
t0 = time.time()
for rnd in range(int(os.environ.get("ROUNDS", 4))):
    pk, ypk, x0, fw = measure(q, cur); own = (x0, max(EPSF * fw, EPSMIN)); q = S.remap0(q, cur, own, shape, DEV); cur = own; F.MAP = cur
    G = S.peak_grid(q[0].item(), ypk, cur, n0, h=h, R=RR, device=DEV)
    r0 = S.residual(q, G, shape).abs().max().item()
    q, hist = S.gn(q, G, shape, free=(True, False), max_iter=30, verbose=bool(os.environ.get("VERBOSE")))
    F.MAP = cur; sup = S.residual(q, G, shape).abs().max().item(); del G; torch.cuda.empty_cache()
    pk, ypk, _, _ = measure(q, cur)
    G2 = S.peak_grid(q[0].item(), ypk, cur, n0, h=h * 0.8, R=RR * 1.3, device=DEV, shift=0.37, widen=1.4); F.MAP = cur
    off = S.residual(q, G2, shape).abs().max().item(); del G2; torch.cuda.empty_cache(); c = q[2:].abs()
    print(f"round {rnd} n0 {n0} h {h} start {r0:.1e} -> lam {q[0].item():.14f} sup {sup:.1e} OFF {off:.1e} its {len(hist)} peak {pk:.5f}@{ypk:.5f} U'(0) {(q[0].item()+2)/2:.12f} eps {cur[1]:.6f} coef", " ".join(f"{c[k:k+n0//8].max().item():.0e}" for k in range(0, n0, n0 // 8)), f"t={time.time()-t0:.0f}s", flush=True)
    torch.save(dict(lam=q[0].item(), p=q.cpu(), map=cur, ypk=ypk, njJ=(1, 0)), A[2])
open("runs/ledger.jsonl", "a").write(json.dumps(dict(case="ccf-smooth-pin2", src=A[1], n0=n0, quad_h=h, lam=q[0].item(), res_sup=sup, res_offgrid=off, peak=pk, y_peak=ypk, map=list(cur), rows_per_fn=RR)) + "\n")
