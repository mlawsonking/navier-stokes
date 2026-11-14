"""Quadrature check against the exact CLM pair: Theta = -ln(1+y^2), Omega = -2y/(1+y^2),
U = 2 atan(y), U' = 2/(1+y^2), U'(0) = 2. (No tail correction: Omega ~ 1/y here, so L is taken large.)"""
import math, torch
from gpu_solver.ccf import Grid
for h in (0.2, 0.1, 0.05):
    G = Grid(h=h, L=45.0, Lc=8.0)
    Th = -torch.log1p(G.ys ** 2); Om = -2 * G.ys / (1 + G.ys ** 2)
    U = (2 * G.y / math.pi) * (G.K @ Th); Up = (2 / math.pi) * (G.K @ (Om * G.ys))
    ratio = -2 / (1 + G.ys ** 2); Up0 = -(2 / math.pi) * (G.w @ ratio)
    print(f"h={h}: U err {(U - 2*torch.atan(G.y)).abs().max().item():.2e}  U' err {(Up - 2/(1+G.y**2)).abs().max().item():.2e}  U'(0) err {abs(Up0.item()-2):.2e}  nodes {G.ys.numel()}")
