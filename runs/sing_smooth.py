"""Pin a smooth CCF profile from the family track: mu = 2 fixed, lam free.
Usage: sing_smooth.py track.pt lam_near n0 h out.pt [nj J]   (env DEV, EPSF, CLUSTER, CW)"""
import sys, os, json, math, time, torch
from gpu_solver import ccf_sing as S, ccf_fast as F
A = sys.argv; src, lam_near, n0, h, outp = A[1], float(A[2]), int(A[3]), float(A[4]), A[5]
nj, J = (int(A[6]), int(A[7])) if len(A) > 6 else (8, 4)
DEV = os.environ.get("DEV", "cuda"); EPSF = float(os.environ.get("EPSF", 1.0)); CL = float(os.environ.get("CLUSTER", 0.9)); CW = float(os.environ.get("CW", 0.5))
L = torch.load(src); L = L if isinstance(L, list) else [L]
x = min(L, key=lambda t: abs(t["lam"] - lam_near)); p = x["p"].to(DEV); cur = tuple(x["map"])
njJ_old = x.get("njJ", (8, 4)); nold = p.numel() - 2 - njJ_old[0] * njJ_old[1]
sing = p[2 + nold:].reshape(njJ_old[1], njJ_old[0]) if njJ_old[1] else torch.zeros(0, 0, dtype=S.DT, device=DEV)
new_sing = torch.zeros(J, nj, dtype=S.DT, device=DEV)
k1, k2 = min(J, sing.shape[0]), min(nj, sing.shape[1] if sing.numel() else 0)
if k1 and k2: new_sing[:k1, :k2] = sing[:k1, :k2]
p = torch.cat([p[:2 + min(nold, n0)], torch.zeros(max(0, n0 - nold), dtype=S.DT, device=DEV), new_sing.reshape(-1)])
shape = (n0, nj, J)
yy = torch.exp(torch.linspace(math.log(0.3), math.log(60.0), 120001, dtype=S.DT, device=DEV))
def measure(p):
    al = 1 / (1 + p[0].item()); Om = (S._fields(p[0], p[1], yy, shape)[1] @ p[2:]).abs()
    i = Om.argmax(); pk = Om[i].item(); hw = (Om > pk / 2).nonzero().flatten(); sf = lambda v: (1 + v * v) ** (-al / 2)
    return pk, yy[i].item(), 2 * sf(yy[i].item()) - 1, 2 * (sf(yy[hw[0]].item()) - sf(yy[hw[-1]].item()))
F.MAP = cur
for rnd in range(3):
    pk, ypk, x0, fw = measure(p)
    new = (x0, max(EPSF * fw, 0.002)); p = S.remap0(p, cur, new, shape, DEV); cur = new; F.MAP = cur
    G = S.EGrid(h=h, device=DEV, cluster=("box", math.log(ypk), 0.15, 40.0) if CL else None)
    p[1] = 2.0; t0 = time.time()
    q, hist = S.lm(p, G, shape, free=(True, False), max_iter=200)
    r = S.residual(q, G, shape); c = q[2:2 + n0].abs(); pk, ypk, _, fw = measure(q)
    print(f"round {rnd} n0 {n0} h {h} lam {q[0].item():.14f} sup {r.abs().max().item():.1e} its {len(hist)} peak {pk:.5f}@{ypk:.4f} U'(0) {(q[0].item()+2)/2:.12f} sing {q[2+n0:].abs().max().item() if nj*J else 0:.1e} coef", " ".join(f"{c[k:k+n0//8].max().item():.0e}" for k in range(0, n0, n0 // 8)), f"{time.time()-t0:.0f}s", flush=True)
    p = q; del G; torch.cuda.empty_cache() if DEV == "cuda" else None
torch.save(dict(lam=p[0].item(), p=p.cpu(), map=cur, njJ=(nj, J)), outp)
open("runs/ledger.jsonl", "a").write(json.dumps(dict(case="ccf-smooth-from-family", n0=n0, nj=nj, J=J, quad_h=h, lam=p[0].item(), res_sup=r.abs().max().item(), peak=pk, y_peak=ypk, map=list(cur), cluster=CL)) + "\n")
