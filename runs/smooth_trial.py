import sys, json, time
sys.path.insert(0, ".")
import torch, run_rung1 as RR
from gpu_solver import burgers as B
from gpu_solver.lm import levenberg_marquardt
DT, dev = torch.float64, "cuda"
y_near = torch.linspace(0.02, 2.0, 40, dtype=DT, device=dev)
for n in (48, 96):
    for (lam0, u3) in ((0.30, True), (0.40, False), (0.55, False)):
        for seed in (0, 1):
            torch.manual_seed(seed)
            y = B.collocation_y(2 * n, device=dev)
            p0 = torch.cat([torch.tensor([lam0], dtype=DT, device=dev), 1e-2 * torch.randn(n, dtype=DT, device=dev)])
            t0 = time.time()
            p, h = levenberg_marquardt(lambda q: B.residual_vec_smooth(q, y, n, u3, y_near), p0, max_iter=1000)
            torch.cuda.synchronize(); wall = time.time() - t0
            lam = p[0].item(); i_br = max(1, round(1 / (2 * lam))); lb = 1 / (2 * i_br)
            yf = torch.linspace(1e-6, 20, 4001, dtype=DT, device=dev)
            U, Uy, s = B.profile(p, yf, n)
            prof = (U - B.exact_U(yf, 2 * i_br + 1)).abs().max().item()
            res = (-p[0] * U + ((1 + p[0]) * yf + U) * Uy).abs().max().item()
            ref = 0.25 if u3 else 0.5
            RR.log(dict(case="1a-burgers-smooth" + ("-U3zero" if u3 else ""), n_basis=n, n_colloc=2 * n,
                        deriv_rows="k=1..6 at 40 pts y in [0.02,2]", seed=seed, lam0=lam0, lam=lam, lam_ref=ref,
                        lam_err=abs(lam - ref), nearest_branch_lam=lb, nearest_branch_lam_err=abs(lam - lb),
                        nearest_branch_profile_err=prof, residual_sup_absy_le_20=res, iters=len(h),
                        wall_s=round(wall, 3), pass_=abs(lam - ref) < 1e-6 and prof < 1e-6 and lb == ref))
