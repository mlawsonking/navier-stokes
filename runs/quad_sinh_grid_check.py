"""EGrid 'sinh' cluster: quadrature check on exact Hilbert pairs with very sharp peaks (see quad_sharp_pair_check)."""
import torch, math
from gpu_solver import ccf_sing as S
for y0, wid in ((7.2, 0.005), (7.2, 0.0005), (1.0, 0.002)):
    z0 = complex(y0, -wid); xp = math.log(y0); e = wid / y0
    def pair(y):
        z = y.to(torch.complex128); Gz = 1j / (z - z0) + 1j / (z + z0.conjugate()); return Gz.real, -Gz.imag
    for name, h, cl in [("box c.15 D200", 0.0125, ("box", xp, 0.15, 200.0)), ("sinh D=0.3/e", 0.0125, ("sinh", xp, e, 0.3 / e)), ("sinh D=0.1/e", 0.0125, ("sinh", xp, e, 0.1 / e)), ("sinh 3e, D=.3/e", 0.0125, ("sinh", xp + e, 3 * e, 0.3 / e))]:
        G = S.EGrid(h=h, device="cuda", cluster=cl); Up_ex, _ = pair(G.y); _, Om_s = pair(G.ys)
        Up = (2 / math.pi) * (G.K @ (Om_s * G.ys)); er = (Up - Up_ex).abs()
        print(f"peak y0 {y0} width {wid}: {name:16s} rows {G.y.numel():6d} sources {G.ys.numel():6d} max|U' err| {er.max().item():.1e} (max|U'| {Up_ex.abs().max().item():.0f}) y range {G.y[0].item():.1e}..{G.y[-1].item():.1e}")
        del G; torch.cuda.empty_cache()
