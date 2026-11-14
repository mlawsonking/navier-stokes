"""Stable self-similar branch of the generalised CLM family omega_t + a u omega_x = u_x omega, continued in a
from the CCF profile (a = -1). Same solver as rung 2a (ccf_fast, lam unknown, origin row lam = (1-a)U'(0) - 2).
Usage: gclm_branch.py n h a_end step device tag"""
import sys, json, time, torch
from gpu_solver import ccf as C, ccf_fast as F
n, h, a_end, step, dev, tag = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5], sys.argv[6]
G = C.Grid(h=h, L=40.0, Lc=22.0, device=dev)
g = torch.load("runs/ccf_stable_h025_n96.pt")[0].to(dev)
p = torch.zeros(n + 1, dtype=C.DT, device=dev); m = min(n, 96); p[:m + 1] = g[:m + 1]
out = open("runs/gclm_branch.jsonl", "a"); a = -1.0; t0 = time.time(); sols = []
sgn = 1.0 if a_end > a else -1.0
while (a - a_end) * sgn <= 1e-9:
    q, hist = F.lm(p, G, a=a, max_iter=200)
    r = F.residual(q[0], q[1:], G, a)
    sup = r.abs().max().item()
    u0 = (q[0].item() + 2) / (1 - a)
    print(f"a {a:+.3f} lam {q[0].item():.13f} sup {sup:.1e} U'(0) {u0:.10f} its {len(hist)} |c_last3| {q[-3:].abs().max().item():.1e} t={time.time()-t0:.0f}s", flush=True)
    out.write(json.dumps(dict(case="gclm-stable-branch-" + tag, a=a, n_basis=n, quad_h=h, lam=q[0].item(), res_sup=sup, iters=len(hist),
                              c_tail=q[-3:].abs().max().item())) + "\n"); out.flush()
    sols.append(torch.cat([torch.tensor([a], dtype=C.DT, device=dev), q]))
    if sup > 1e-6:
        print("stopping: not converged"); break
    p = q; a = round(a + sgn * step, 10)
torch.save(torch.stack(sols), f"runs/gclm_branch_{tag}.pt")
