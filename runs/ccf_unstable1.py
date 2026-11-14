"""First unstable CCF profile. Normalisation Omega'(0) = -d (d = 0.1 puts the peak near y = 4, as in Fig. 3 of
arXiv:2201.06780, which was used ONLY to shape the initial guess). Stage 1: lam fixed at lam_init; stage 2: lam free.
Usage: ccf_unstable1.py n h d lam_init device [out.pt]"""
import sys, time, torch
from gpu_solver import ccf as C, ccf_fast as F
n, h, d, lam0, dev = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
F.MODE.update(norm='origin', origin_row=True, d=d)
G = C.Grid(h=h, L=40.0, Lc=22.0, device=dev)
lam = torch.tensor(lam0, dtype=C.DT, device=dev)
al = 1 / (1 + lam0)
y = G.y
k, y0, q = (1.6 - 4 * d) / 64, 4.0, 4
Og = -(d * y + k * y ** 3) / (1 + (y / y0) ** (2 * q)) ** ((3 + al) / (2 * q))
BOm = F.operators(lam, G, n)[0]
wts = 1 / (y * (1 + y * y) ** (-(1 + al) / 2) + 1e-3)
c = torch.linalg.lstsq(BOm * wts[:, None], (Og * wts)[:, None]).solution[:, 0]
print("guess fit rel err", ((BOm @ c - Og).abs().max() / Og.abs().max()).item())
p = torch.cat([lam.reshape(1), c])
for free in (False, True):
    t0 = time.time()
    p, hist = F.lm(p, G, lam_free=free, max_iter=400)
    r = F.residual(p[0], p[1:], G)
    ops = F.operators(p[0], G, n)
    print(f"lam_free={free}: lam {p[0].item():.13f} sup {r.abs().max().item():.2e} l2 {r.norm().item():.2e} its {len(hist)} "
          f"U'(0) {(ops[4] @ p[1:]).item():.8f} Om'(0) {(ops[5] @ p[1:]).item():.5f} |c| mid {p[n//2-1:n//2+2].abs().max().item():.1e} last {p[-3:].abs().max().item():.1e}  {time.time()-t0:.0f}s", flush=True)
Om = ops[0] @ p[1:]
i = Om.abs().argmax()
print(f"peak |Omega| = {Om[i].abs().item():.4f} at y = {G.y[i].item():.3f}")
torch.save(p, sys.argv[6] if len(sys.argv) > 6 else f"runs/ccf_u1_n{n}.pt")
