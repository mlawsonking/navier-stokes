"""Diagnostic (HANDOFF Q1): lam fixed on a grid, solve coefficients only (old rows: collocation + sum c = 1).
All grid values solved in one batched LM on the GPU. Prediction: a V centred at 0.25, flatter as n grows."""
import json, time, torch
from gpu_solver import burgers as B
from gpu_solver.batch_lm import batched_lm
dev, DT = "cuda", torch.float64
lams = torch.linspace(0.24, 0.26, 21, dtype=DT, device=dev)
out = open("runs/lam_scan.jsonl", "a")
for n in (48, 96):
    y = B.collocation_y(2 * n, device=dev)
    f = lambda c, lam: B.residual_vec(torch.cat([lam.reshape(1), c]), y, n, u3_zero=True)
    torch.manual_seed(0)
    c0 = 1e-2 * torch.randn(21, n, dtype=DT, device=dev)
    t0 = time.time()
    c, sup, it = batched_lm(f, c0, aux=lams, max_iter=1000)
    torch.cuda.synchronize(); wall = time.time() - t0
    r = torch.func.vmap(f)(c, lams)
    l2 = r.norm(dim=1)
    print(f"n={n} wall {wall:.1f}s")
    for L, a, b_, i in zip(lams.tolist(), l2.tolist(), sup.tolist(), it.tolist()):
        print(f"  lam {L:.3f}  |r|_2 {a:.3e}  sup {b_:.3e}  it {i}")
        out.write(json.dumps(dict(case="1a-lam-scan", n_basis=n, lam=L, res_l2=a, res_sup=b_, iters=i, wall_s_batch=round(wall, 2))) + "\n")
