"""First unstable gCLM branch in a with the log-regular representation (crosses lam = 0).
Usage: gclm_log_branch.py src.pt a_end step tag [n]    src: gclm_u1_branch list (ccf_fast format) or a list
written by this script (resume). env: DEV, THREADS, EPSF, H"""
import sys, os, json, math, time, torch
from gpu_solver import ccf as C, ccf_fast as F, gclm_log as L, ccf_sing as S
A = sys.argv; src, a_end, step, tag = A[1], float(A[2]), float(A[3]), A[4]
DEV = os.environ.get("DEV", "cpu"); torch.set_num_threads(int(os.environ.get("THREADS", 8)))
EPSF = float(os.environ.get("EPSF", 0.66)); h = float(os.environ.get("H", 0.0125)); L.D = 0.1; CC = float(os.environ.get("CC", 0.3)); CD = float(os.environ.get("CD", 20)); RR = float(os.environ.get("ROWS", 6))
yy = torch.linspace(0.3, 14.0, 137001, dtype=C.DT, device=DEV)

def nodes(n, mp):
    j = torch.arange(n, dtype=C.DT, device=DEV); t = torch.cos(math.pi * (j + 0.5) / n)
    x0, eps = mp; b1, b2 = math.asinh((1 - x0) / eps), math.asinh((1 + x0) / eps)
    return (x0 + eps * torch.sinh((b1 + b2) * (t + 1) / 2 - b2) + 1) / 2

def measure(p, mp):
    F.MAP = mp; n = p.numel() - 2
    B = L._basis(p[0], yy, n); Om = (B[1] @ p[1:]).abs(); s = B[6]
    i = Om.argmax(); pk = Om[i].item(); hw = (Om > pk / 2).nonzero().flatten()
    return pk, yy[i].item(), 2 * s[i].item() - 1, 2 * abs(s[hw[0]].item() - s[hw[-1]].item())

def remap(p, old, new):
    n = p.numel() - 2; s = nodes(n, new)
    F.MAP = old; N = F.basis3(s, n)[0] @ p[2:]
    F.MAP = new; return torch.cat([p[:2], torch.linalg.solve(F.basis3(s, n)[0], N)])

def align(p, mp):
    pk, ypk, x0, fw = measure(p, mp); own = (x0, max(EPSF * fw, 0.0002))
    return remap(p, mp, own), own, ypk, pk

S0 = torch.load(src); states = []
for x in S0[-3:]:
    mp = tuple(x["map"]); p = x["p"].to(DEV)
    if "ypk" not in x:                                 # ccf_fast format -> log format
        F.MAP = mp; n = p.numel() - 1; p = L.from_fast(p, n, nodes(n, mp))
    if len(A) > 5 and int(A[5]) > p.numel() - 2: p = torch.cat([p, torch.zeros(int(A[5]) - p.numel() + 2, dtype=C.DT, device=DEV)])
    p, mp, ypk, _ = align(p, mp); states.append((p, mp, ypk, x["a"]))
sols = S0 if "ypk" in S0[-1] else []
sgn = 1.0 if a_end > states[-1][3] else -1.0; a = round(states[-1][3] + sgn * step, 10); step0 = step
out = open("runs/gclm_u1_branch.jsonl", "a"); t0 = time.time()
while (a - a_end) * sgn <= 1e-12:
    p, mp, ypk, ap = states[-1]
    use = states[-3:]; ls = [st[3] for st in use]
    wts = [math.prod((a - ls[k]) / (ls[i] - ls[k]) for k in range(len(use)) if k != i) for i in range(len(use))]
    guess = sum(w_ * st[0] for w_, st in zip(wts, use))
    cur = (sum(w_ * st[1][0] for w_, st in zip(wts, use)), math.exp(sum(w_ * math.log(st[1][1]) for w_, st in zip(wts, use))))
    cur = (cur[0], min(max(cur[1], 0.5 * mp[1]), 2 * mp[1])); yg = math.exp(sum(w_ * math.log(st[2]) for w_, st in zip(wts, use)))
    F.MAP = cur
    G = S.peak_grid(guess[0].item(), yg, cur, guess.numel() - 2, h=h, R=RR, device=DEV)   # exp grid, rows follow the basis density
    q, hist = L.gn(guess, G, a=a, max_iter=25)
    sup = L.residual(q[0], q[1:], G, a).abs().max().item()
    G2 = S.peak_grid(guess[0].item(), yg, cur, guess.numel() - 2, h=h * 0.8, R=RR * 1.3, device=DEV, shift=0.37, widen=1.4)
    off = L.residual(q[0], q[1:], G2, a).abs().max().item(); del G, G2
    # the terms of the residual are of size peak^2/width, so the rounding floor grows with the peak:
    # accept on the residual RELATIVE to that scale (and still require on-grid = off-grid within 10x)
    pk0, _, _, fw0 = measure(q, cur); scale = max(1.0, pk0 * pk0 / max(fw0 / 2, 1e-12) * 1e-3)
    if not (off / scale < float(os.environ.get("ACCEPT", 1e-7)) and off < 10 * sup + 1e-9):
        a = round(a - sgn * step, 10); step = step / 2; a = round(a + sgn * step, 10)
        print(f"  rejected (sup {sup:.1e} off {off:.1e} its {len(hist)}); step -> {step}", flush=True)
        if step < 1e-5: print("stopping: step too small"); break
        continue
    q, own, yq_, pk = align(q, cur); c = q[2:].abs(); n = c.numel()
    print(f"a {a:+.5f} lam {q[0].item():+.12f} q0 {q[1].item():+.8f} sup {sup:.1e} OFF {off:.1e} rel {off/scale:.1e} its {len(hist)} peak {pk:.4f}@{yq_:.3f} eps {own[1]:.4f} c_mid {c[n//2-2:n//2+1].max().item():.0e} c_last {c[-3:].max().item():.0e} t={time.time()-t0:.0f}s", flush=True)
    out.write(json.dumps(dict(case="gclm-u1-log-" + tag, a=a, n_basis=n, quad_h=h, lam=q[0].item(), q0=q[1].item(), res_sup=sup, res_offgrid=off, res_rel=off / scale, iters=len(hist), peak=pk, y_peak=yq_, map=list(own))) + "\n"); out.flush()
    states.append((q, own, yq_, a)); states = states[-3:]
    sols.append(dict(a=a, p=q.clone().cpu(), map=own, ypk=yq_)); torch.save(sols, f"runs/gclm_log_branch_{tag}.pt")
    step = min(step * 1.5, step0); a = round(a + sgn * step, 10)
