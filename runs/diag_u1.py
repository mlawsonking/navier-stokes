import sys, torch, math
from gpu_solver import ccf as C, ccf_fast as F
sols = torch.load("runs/gclm_u1_branch_up192d.pt")
F.MODE.update(norm="origin", origin_row=True, d=0.1, ymin=0.0)
G = C.Grid(h=0.0125, L=40.0, Lc=22.0, device="cpu", n_gl=128)
for s in sols:
    if abs(s['a'] - float(sys.argv[1])) > 1e-9: continue
    p = s['p']; F.MAP = tuple(s['map']); a = s['a']
    r = F.residual(p[0], p[1:], G, a)
    c = p[1:].abs()
    print("a", a, "lam", p[0].item(), "nres", r.numel())
    print("coef decay:", " ".join(f"{c[i:i+16].max().item():.0e}" for i in range(0, c.numel(), 16)))
    y = G.y; m = y.numel()
    rr = r[:m].abs()
    edges = [0, 0.01, 0.1, 0.5, 1, 2, 3, 3.5, 4, 5, 8, 20, 100, 1e4, 1e8, 1e20]
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (y >= lo) & (y < hi)
        if k.any(): print(f"y in [{lo},{hi}): n={int(k.sum())} max res {rr[k].max().item():.1e}")
    print("extra rows", r[m:])
