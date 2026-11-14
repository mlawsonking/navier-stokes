import torch, math, time, sys
from gpu_solver import ccf_sing as S, ccf_fast as F
DEV = "cuda"; x = torch.load("runs/sing_track2_v1.pt")[-1]; p = x["p"].to(DEV); F.MAP = tuple(x["map"]); shape = (640, 8, 4)
G = S.EGrid(h=0.0125, device=DEV, cluster=("box", math.log(x["ypk"]), 0.15, 80.0))
r = S.residual(p, G, shape); print("at solution: sup", r.abs().max().item())
r, J = S.res_jac(p, G, shape, (False, True))
d = J.norm(dim=0); sv = torch.linalg.svdvals(J / d)
print("scaled Jacobian singular values: max %.2e min %.2e; smallest 8:" % (sv[0].item(), sv[-1].item()), [f"{v:.1e}" for v in sv[-8:].tolist()])
# FD check of the mu column
for e in (1e-4, 1e-6, 1e-8):
    q1 = p.clone(); q1[1] += e; q2 = p.clone(); q2[1] -= e
    col = (S.residual(q1, G, shape) - S.residual(q2, G, shape)) / (2 * e); print("mu column, e =", e, "norm", col.norm().item(), "diff to e=1e-6 col", (col - J[:, 0]).norm().item())
g = p.clone(); g[0] = p[0] - float(sys.argv[1])
r0 = S.residual(g, G, shape); print("start (lam shifted by", sys.argv[1], ") sup", r0.abs().max().item(), "l2", r0.norm().item())
# plain Gauss-Newton steps with step-halving, to see the raw behaviour
q = g.clone(); idx = torch.arange(1, q.numel(), device=DEV)
for it in range(12):
    r, J = S.res_jac(q, G, shape, (False, True)); d = J.norm(dim=0)
    st = torch.linalg.lstsq(J / d, -r[:, None]).solution[:, 0] / d
    t = 1.0
    while t > 1e-4:
        qn = q.clone(); qn[idx] += t * st; rn = S.residual(qn, G, shape)
        if rn.norm() < r.norm(): break
        t /= 2
    q = qn; print(f"GN it {it} damping t {t:.3f} l2 {rn.norm().item():.2e} sup {rn.abs().max().item():.2e} mu {q[1].item():.9f} |step| {st.abs().max().item():.1e}", flush=True)
