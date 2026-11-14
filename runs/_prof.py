import torch, time, os
from gpu_solver import ccf_sing as S, ccf_fast as F
DEV = os.environ.get("DEV", "cpu"); torch.set_num_threads(8)
sols = torch.load("runs/sing_track_t2_to054.pt"); a, b = sols[-2], sols[-1]
shape = (192, 8, 4); G = S.EGrid(h=0.0125, device=DEV); F.MAP = tuple(b["map"])
p = b["p"].to(DEV)
def T(f, k=3):
    t = time.time()
    for _ in range(k): f()
    if DEV == "cuda": torch.cuda.synchronize()
    return (time.time() - t) / k
print("residual", T(lambda: S.residual(p, G, shape)), "res_jac", T(lambda: S.res_jac(p, G, shape, (False, True))))
print("fields ys", T(lambda: S._fields(p[0], p[1], G.ys, shape)), "K@B", T(lambda: G.K @ torch.zeros(G.ys.numel(), 224, dtype=S.DT, device=DEV)))
# convergence history from a perturbed lam (step of 0.005 with no predictor)
g = p.clone(); g[0] = g[0] - 0.005
for mu0 in (1e-3, 1e-6, 1e-10):
    t = time.time(); q, h = S.lm(g, G, shape, max_iter=150, mu0=mu0)
    print("mu0", mu0, "its", len(h), "time", round(time.time() - t), "sup", h[-1][1], "mu", q[1].item(), [f"{x[1]:.0e}" for x in h[::4]])
