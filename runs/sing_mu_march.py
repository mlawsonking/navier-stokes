"""March the non-smooth CCF family in mu (lam UNKNOWN) down to mu = 2, i.e. onto a smooth profile.
Usage: sing_mu_march.py track2.pt out.pt n0 h dmu [mu_end=2]   env: EPSF, EPSMIN, ROWS, ACCEPT
Same peak-aligned Lagrange predictor and peak_grid as sing_track2.py; the roles of lam and mu are swapped."""
import sys, os, json, math, time, torch
from gpu_solver import ccf_sing as S, ccf_fast as F
A = sys.argv; src, outp, n0, h, dmu = A[1], A[2], int(A[3]), float(A[4]), float(A[5]); mu_end = float(A[6]) if len(A) > 6 else 2.0
DEV = "cuda"; EPSF = float(os.environ.get("EPSF", 1.0)); EPSMIN = float(os.environ.get("EPSMIN", 4e-5)); RR = float(os.environ.get("ROWS", 6)); ACC = float(os.environ.get("ACCEPT", 1e-6))
sols = torch.load(src); nj, J = sols[-1]["njJ"]; shape = (n0, nj, J)
yy = torch.exp(torch.linspace(math.log(0.3), math.log(60.0), 480001, dtype=S.DT, device=DEV))
def measure(p, mp):
    F.MAP = mp; al = 1 / (1 + p[0].item()); Om = (S._fields(p[0], p[1], yy, shape)[1] @ p[2:]).abs()
    i = Om.argmax(); pk = Om[i].item(); hw = (Om > pk / 2).nonzero().flatten(); sf = lambda v: (1 + v * v) ** (-al / 2)
    return pk, yy[i].item(), 2 * sf(yy[i].item()) - 1, 2 * (sf(yy[hw[0]].item()) - sf(yy[hw[-1]].item()))
def align(p, mp):
    pk, ypk, x0, fw = measure(p, mp); own = (x0, max(EPSF * fw, EPSMIN)); return S.remap0(p, mp, own, shape, DEV), own, ypk, pk
def padto(v):
    nold = v.numel() - 2 - nj * J
    return torch.cat([v[:2 + min(nold, n0)], torch.zeros(max(0, n0 - nold), dtype=S.DT, device=DEV), v[2 + nold:]])
states = [(padto(x["p"].to(DEV)), tuple(x["map"]), x["ypk"]) for x in sols[-3:]]
out = []; t0 = time.time(); step0 = dmu; mu = max(states[-1][0][1].item() - dmu, mu_end)
while True:
    use = states[-3:]; ms = [st[0][1].item() for st in use]; p, mp, ypk = states[-1]
    wts = [math.prod((mu - ms[k]) / (ms[i] - ms[k]) for k in range(len(use)) if k != i) for i in range(len(use))]
    guess = sum(w_ * st[0] for w_, st in zip(wts, use)); guess[1] = mu
    cur = (sum(w_ * st[1][0] for w_, st in zip(wts, use)), math.exp(sum(w_ * math.log(st[1][1]) for w_, st in zip(wts, use))))
    cur = (cur[0], min(max(cur[1], 0.5 * mp[1]), 2 * mp[1])); yg = math.exp(sum(w_ * math.log(st[2]) for w_, st in zip(wts, use)))
    F.MAP = cur; G = S.peak_grid(guess[0].item(), yg, cur, n0, h=h, R=RR, device=DEV)
    q, hist = S.gn(guess, G, shape, free=(True, False), max_iter=25)
    F.MAP = cur; sup = S.residual(q, G, shape).abs().max().item(); del G; torch.cuda.empty_cache()
    if not sup < ACC:
        dmu = dmu / 2; mu = max(states[-1][0][1].item() - dmu, mu_end)
        print(f"  rejected mu {mu + dmu:.6f} (sup {sup:.1e}, its {len(hist)}); dmu -> {dmu:.2e}", flush=True)
        if dmu < 1e-6: print("stopping: step too small"); break
        continue
    q, own, yq_, pk = align(q, cur); F.MAP = own
    G2 = S.peak_grid(q[0].item(), yq_, own, n0, h=h * 0.8, R=RR * 1.3, device=DEV, shift=0.37, widen=1.4); off = S.residual(q, G2, shape).abs().max().item(); del G2; torch.cuda.empty_cache()
    c = q[2:2 + n0].abs()
    print(f"mu {mu:.6f} lam {q[0].item():.12f} U'(0) {(1 + q[0].item()) - q[0].item() / mu:.10f} sup {sup:.1e} OFF {off:.1e} its {len(hist)} peak {pk:.4f}@{yq_:.4f} eps {own[1]:.5f} c_mid {c[n0//2-2:n0//2+1].max().item():.0e} c_last {c[-3:].max().item():.0e} t={time.time()-t0:.0f}s", flush=True)
    open("runs/sing_track.jsonl", "a").write(json.dumps(dict(case="sing-mu-march", mu=mu, lam=q[0].item(), shape=list(shape), quad_h=h, res_sup=sup, res_offgrid=off, iters=len(hist), peak=pk, y_peak=yq_, map=list(own))) + "\n")
    states.append((q, own, yq_)); out.append(dict(lam=q[0].item(), mu=mu, p=q.clone().cpu(), map=own, ypk=yq_, njJ=(nj, J))); torch.save(out, outp)
    if mu <= mu_end + 1e-15: break
    dmu = min(dmu * 1.5, step0); mu = max(mu - dmu, mu_end)
