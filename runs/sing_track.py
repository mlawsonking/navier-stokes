"""Track the non-smooth CCF family in lam (mu free) from the first unstable profile.
Usage: sing_track.py lam_end step tag [n0 nj J h]   (env DEV, EPSF, SRC=previous track .pt to resume)"""
import sys, os, json, math, time, torch
from gpu_solver import ccf_sing as S, ccf_fast as F
A = sys.argv; lam_end, step, tag = float(A[1]), float(A[2]), A[3]
n0, nj, J, h = (int(A[4]), int(A[5]), int(A[6]), float(A[7])) if len(A) > 4 else (192, 8, 4, 0.0125)
shape = (n0, nj, J); DEV = os.environ.get("DEV", "cuda"); EPSF = float(os.environ.get("EPSF", 0.66))
G = S.EGrid(h=h, device=DEV); CL = float(os.environ.get('CLUSTER', 0.9))
yy = torch.exp(torch.linspace(math.log(0.3), math.log(30.0), 60001, dtype=S.DT, device=DEV))

def measure(p, mp):
    F.MAP = mp; al = 1 / (1 + p[0].item())
    Om = S._fields(p[0], p[1], yy, shape)[1] @ p[2:]
    i = Om.abs().argmax(); pk = Om[i].abs().item(); hw = (Om.abs() > pk / 2).nonzero().flatten()
    sf = lambda v: (1 + v * v) ** (-al / 2)
    return pk, yy[i].item(), 2 * sf(yy[i].item()) - 1, 2 * (sf(yy[hw[0]].item()) - sf(yy[hw[-1]].item()))

if os.environ.get("SRC"):
    sols = torch.load(os.environ["SRC"]); last = sols[-1]; p = last["p"].to(DEV); cur = tuple(last["map"])
    prev = sols[-2]["p"].to(DEV); prevmap = tuple(sols[-2]["map"]); lam = round(p[0].item() - step, 10)
    nold = p.numel() - 2 - nj * J                       # resume at a larger n0: zero-pad the N_0 block (same map)
    pad = lambda v: torch.cat([v[:2 + nold], torch.zeros(n0 - nold, dtype=S.DT, device=DEV), v[2 + nold:]])
    p, prev = pad(p), pad(prev)
else:
    old = (-0.14, 0.1); p, _ = S.from_smooth(torch.load("runs/ccf_u1_v_160_0.0125_0.1.pt").reshape(-1), old, old, shape, DEV)
    cur = old; sols = []; prev = None; lam = round(math.floor(p[0].item() / step) * step, 10)
out = open("runs/sing_track.jsonl", "a"); t0 = time.time()
while lam >= lam_end - 1e-12:
    pk, ypk, x0, fw = measure(p, cur)
    new = (x0, max(EPSF * fw, 0.004)); p = S.remap0(p, cur, new, shape, DEV); cur = new
    guess = p.clone()
    if CL: del G; torch.cuda.empty_cache(); G = S.EGrid(h=h, device=DEV, cluster=(math.log(ypk), 0.5, CL * 0.5))
    if prev is not None:
        pr = S.remap0(prev, prevmap, cur, shape, DEV); f = (lam - p[0].item()) / (p[0].item() - pr[0].item())
        guess = p + f * (p - pr)
    guess[0] = lam
    q, hist = S.lm(guess, G, shape, free=(False, True), max_iter=60)
    F.MAP = cur; r = S.residual(q, G, shape); sup = r.abs().max().item()
    u0 = (1 + lam) - lam / q[1].item()
    pk, ypk, _, fw = measure(q, cur); c = q[2:2 + n0].abs(); cs = q[2 + n0:].abs()
    print(f"lam {lam:.4f} mu {q[1].item():.10f} U'(0) {u0:.8f} sup {sup:.1e} its {len(hist)} peak {pk:.4f}@{ypk:.3f} fw {fw:.4f} c_mid {c[n0//2-2:n0//2+1].max().item():.0e} c_last {c[-3:].max().item():.0e} sing {cs.max().item() if cs.numel() else 0:.1e} t={time.time()-t0:.0f}s", flush=True)
    out.write(json.dumps(dict(case="sing-track-" + tag, lam=lam, mu=q[1].item(), up0=u0, shape=list(shape), quad_h=h, res_sup=sup, iters=len(hist), peak=pk, y_peak=ypk, map=list(cur))) + "\n"); out.flush()
    if sup > float(os.environ.get("ACCEPT", 3e-8)):          # reject: retry from the last good point with half the step
        lam = round(lam + step, 10); step = step / 2; lam = round(lam - step, 10)
        print(f"  rejected (sup {sup:.1e}); step -> {step}", flush=True)
        if step < 2e-4: print("stopping: step too small"); break
        continue
    prev, prevmap, p = p, cur, q
    sols.append(dict(lam=lam, p=q.clone().cpu(), map=cur)); torch.save(sols, f"runs/sing_track_{tag}.pt")
    lam = round(lam - step, 10)
