"""Can the basis represent the exact profiles? Fit c (lam fixed) to exact U by linear least squares."""
import sys; sys.path.insert(0, ".")
import torch
from gpu_solver import burgers as B
DT, dev = torch.float64, "cuda"
yf = torch.linspace(1e-4, 20, 4001, dtype=DT, device=dev)
for lam, m in ((0.5, 3), (0.25, 5)):
    a = 1 / (1 + lam)
    for n in (12, 24, 48, 96):
        y = torch.cat([B.collocation_y(4 * n, lam_ref=lam, device=dev), yf])
        s = (1 + y * y) ** (-a / 2)
        T, _ = B.cheb_T_dT(2 * s - 1, n)
        Ue = B.exact_U(y, m)
        # U = -y s (1 + q P)  =>  q P = -U/(y s) - 1, linear in c
        A = (s - s * s)[:, None] * T
        rhs = -Ue / (y * s) - 1
        c = torch.linalg.lstsq(A, rhs[:, None]).solution[:, 0]
        p = torch.cat([torch.tensor([lam], dtype=DT, device=dev), c])
        U, Uy, _ = B.profile(p, yf, n)
        R = (-lam * U + ((1 + lam) * yf + U) * Uy).abs().max().item()
        print(f"lam={lam} n={n:3d} best-fit sup|U-Uexact| (|y|<=20) = {(U - B.exact_U(yf, m)).abs().max().item():.2e}  residual = {R:.2e}  |c_last| = {c[-4:].abs().max().item():.1e}")
