"""Quadrature check on an exact Hilbert pair with a sharp peak: G = U' - i Omega = i/(z-z0) + i/(z+conj z0), Im z0 < 0."""
import torch, math
from gpu_solver import ccf as C
for wid in (0.05, 0.02):
    z0 = complex(3.8, -wid)
    def pair(y):
        z = y.to(torch.complex128); Gz = 1j / (z - z0) + 1j / (z + z0.conjugate()); return Gz.real, -Gz.imag
    xp = math.asinh(3.8)
    for name, h, cl in [("plain h", 0.0125, None), ("plain h/2", 0.00625, None), ("clust 20x", 0.0125, (xp, 0.5, 0.475)), ("clust 5x", 0.0125, (xp, 0.5, 0.4)), ("clust 20x h/2", 0.00625, (xp, 0.5, 0.475))]:
        G = C.Grid(h=h, L=40.0, Lc=22.0, device="cpu", n_gl=8, cluster=cl)
        Up_ex, _ = pair(G.y); _, Om_s = pair(G.ys)
        Up = (2 / math.pi) * (G.K @ (Om_s * G.ys))
        e = (Up - Up_ex).abs(); i = e.argmax()
        print(f"width {wid} {name:14s} max |U' error| {e[i].item():.1e} at y {G.y[i].item():.3f}   (max |U'| {Up_ex.abs().max().item():.1f})")
