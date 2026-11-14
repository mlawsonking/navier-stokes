"""Random multi-start in the nu-family at fixed nu. Usage: ccf_nu_multistart.py n h nu nstarts device seed [lam_fixed]"""
import sys, time, torch
from gpu_solver import ccf as C, ccf_family as F
n, h, nu, ns, dev, seed = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5], int(sys.argv[6])
lam_fixed = float(sys.argv[7]) if len(sys.argv) > 7 else None
G = C.Grid(h=h, L=40.0, Lc=22.0, device=dev)
torch.manual_seed(seed)
k = torch.arange(1, n + 1, dtype=C.DT, device=dev)
best, t0 = [], time.time()
for i in range(ns):
    amp = [0.0, 0.3, 1.0, 3.0][i % 4]
    c = amp * torch.randn(n, dtype=C.DT, device=dev) / k ** 2
    c[0] += -1.0 - c.sum()
    lam0 = lam_fixed if lam_fixed is not None else 0.35 + 0.9 * torch.rand(1).item()
    p, hist = F.lm(torch.cat([torch.tensor([lam0], dtype=C.DT, device=dev), c]), nu, G, lam_free=lam_fixed is None, max_iter=200)
    r = F.residual(p[0], p[1:], nu, G)
    u0 = (F.operators(p[0], (nu + 1) / 2, G, n)[4] @ p[1:]).item()
    best.append((r.norm().item(), i, p))
    print(f"start {i} amp {amp} lam0 {lam0:.3f} -> lam {p[0].item():.9f} l2 {r.norm().item():.3e} sup {r.abs().max().item():.2e} U'(0) {u0:.5f} its {len(hist)} c_last {p[-3:].abs().max().item():.1e}", flush=True)
best.sort(key=lambda t: t[0])
print("best:", [(f"{b[0]:.2e}", b[1], round(b[2][0].item(), 6)) for b in best[:6]], f"wall {time.time()-t0:.0f}s")
torch.save(torch.stack([b[2] for b in best[:6]]), f"runs/ccf_nu{nu}_ms_n{n}.pt")
