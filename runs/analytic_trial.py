"""1a with real collocation + complex off-axis collocation (analyticity enforces smoothness at y=0)."""
import sys, time
sys.path.insert(0, ".")
import torch, run_rung1 as RR
from gpu_solver import burgers as B
from gpu_solver.lm import levenberg_marquardt
DT, dev = torch.float64, "cuda"
x = torch.linspace(0.05, 4.0, 40, dtype=DT, device=dev)
# exact profiles have branch points at |Im y| ~ 0.385 (lam=1/2: y=-U-U^3, 1+3U^2=0) and 0.38 (lam=1/4),
# so the complex rows must stay inside that strip
z = torch.cat([x + 0.1j, x + 0.2j]).to(torch.complex128)
ns = [int(a) for a in sys.argv[1:]] or [48, 96]
for n in ns:
    for (lam0, u3) in ((0.30, True), (0.20, True), (0.40, False), (0.55, False)):
        for seed in (0, 1):
            torch.manual_seed(seed)
            y = B.collocation_y(2 * n, device=dev)
            p0 = torch.cat([torch.tensor([lam0], dtype=DT, device=dev), 1e-2 * torch.randn(n, dtype=DT, device=dev)])
            t0 = time.time()
            p1, h1 = levenberg_marquardt(lambda q: B.residual_vec(q, y, n, u3), p0, max_iter=400)
            lam_stage1 = p1[0].item()
            p, h2 = levenberg_marquardt(lambda q: B.residual_vec_analytic(q, y, n, u3, z), p1, max_iter=400)
            h = h1 + h2
            torch.cuda.synchronize(); wall = time.time() - t0
            lam = p[0].item(); i_br = max(1, round(1 / (2 * lam))); lb = 1 / (2 * i_br)
            yf = torch.linspace(1e-6, 20, 4001, dtype=DT, device=dev)
            U, Uy, s = B.profile(p, yf, n)
            prof = (U - B.exact_U(yf, 2 * i_br + 1)).abs().max().item()
            res = (-p[0] * U + ((1 + p[0]) * yf + U) * Uy).abs().max().item()
            resc = B.residual_complex(p, z, n).abs().max().item()
            ref = 0.25 if u3 else 0.5
            RR.log(dict(case="1a-burgers-analytic2" + ("-U3zero" if u3 else ""), n_basis=n, n_colloc=2 * n,
                        complex_rows="80 pts Re in [0.05,4], Im in {0.1,0.2}; stage 1 real-only, stage 2 real+complex", seed=seed, lam0=lam0, lam=lam,
                        lam_stage1=lam_stage1, lam_ref=ref, lam_err=abs(lam - ref), nearest_branch_lam=lb,
                        nearest_branch_lam_err=abs(lam - lb), nearest_branch_profile_err=prof,
                        residual_sup_absy_le_20=res, residual_sup_complex=resc, iters=len(h),
                        wall_s=round(wall, 3), pass_=abs(lam - ref) < 1e-6 and prof < 1e-6))
