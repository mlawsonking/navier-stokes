import torch, math, time, sys
from gpu_solver import ccf_sing as S, ccf_fast as F
DEV = "cuda"; x = torch.load("runs/sing_track2_v1.pt")[-1]; p = x["p"].to(DEV); F.MAP = tuple(x["map"]); shape = (640, 8, 4)
G = S.EGrid(h=0.0125, device=DEV, cluster=("box", math.log(x["ypk"]), 0.15, 80.0))
g = p.clone(); g[0] = p[0] - float(sys.argv[1]); idx = torch.arange(1, g.numel(), device=DEV)
for rc in (1e-9, 1e-6):
    q = g.clone(); t0 = time.time()
    for it in range(14):
        r, J = S.res_jac(q, G, shape, (False, True)); d = J.norm(dim=0)
        U, sv, Vh = torch.linalg.svd(J / d, full_matrices=False); keep = sv > rc * sv[0]
        st = (Vh[keep].T @ ((U[:, keep].T @ -r) / sv[keep])) / d
        t = 1.0
        while t > 1e-3:
            qn = q.clone(); qn[idx] += t * st; rn = S.residual(qn, G, shape)
            if rn.norm() < r.norm(): break
            t /= 2
        q = qn; print(f"TSVD rc {rc:.0e} it {it} t {t:.3f} l2 {rn.norm().item():.2e} sup {rn.abs().max().item():.2e} mu {q[1].item():.10f} |step| {st.abs().max().item():.1e} rank {int(keep.sum())} {time.time()-t0:.0f}s", flush=True)
        if rn.abs().max().item() < 3e-9: break
q, hist = S.lm(g, G, shape, free=(False, True), max_iter=25)
print("LM:", [f"{h_[1]:.1e}" for h_ in hist])
