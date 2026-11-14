"""Continue the CCF family in the origin exponent nu from 1 (stable profile) upward.
Usage: ccf_nu_continuation.py n h nu_max step device"""
import sys, json, time, torch
from gpu_solver import ccf as C, ccf_family as F
n, h, nu_max, step, dev = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
G = C.Grid(h=h, L=40.0, Lc=22.0, device=dev)
p = torch.zeros(n + 1, dtype=C.DT, device=dev); p[0] = 1.2; p[1] = -1.0
out = open("runs/ccf_family.jsonl", "a"); sols = []
nu, t0 = 1.0, time.time()
while nu <= nu_max + 1e-9:
    q, hist = F.lm(p, nu, G, max_iter=300)
    r = F.residual(q[0], q[1:], nu, G)
    u0 = (F.operators(q[0], (nu + 1) / 2, G, n)[4] @ q[1:]).item()
    sup = r.abs().max().item()
    print(f"nu {nu:.3f} lam {q[0].item():.12f} sup {sup:.2e} U'(0) {u0:.9f} its {len(hist)} |c_last3| {q[-3:].abs().max().item():.1e}  t={time.time()-t0:.0f}s", flush=True)
    out.write(json.dumps(dict(case="2a-ccf-nu-family", n_basis=n, quad_h=h, nu=nu, lam=q[0].item(), res_sup=sup, Up0=u0, iters=len(hist),
                              c_tail=q[-3:].abs().max().item())) + "\n"); out.flush()
    sols.append(torch.cat([torch.tensor([nu], dtype=C.DT, device=dev), q]))
    if sup < 1e-3:
        p = q
    nu = round(nu + step, 10)
torch.save(torch.stack(sols), f"runs/ccf_family_n{n}_h{h}.pt")
