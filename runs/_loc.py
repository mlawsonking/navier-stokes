import torch, math
from gpu_solver import ccf as C, ccf_fast as F
F.MODE.update(norm="origin", origin_row=True, d=0.1, ymin=0.0); F.FAR = None
sols = torch.load("runs/gclm_u1_branch_upc288.pt")
x = [t for t in sols if abs(t['a'] + 0.52) < 1e-9][0]; p = x['p']; F.MAP = tuple(x['map']); n = p.numel() - 1; lam = p[0].item(); al = 1/(1+lam)
print("lam", lam, "map", F.MAP)
G = C.Grid(h=0.0125, L=40.0, Lc=22.0, device="cpu", n_gl=128, cluster=None)
r = F.residual(p[0], p[1:], G, -0.52)[:-2].abs(); y = G.y
for lo, hi in [(0,.01),(.01,.1),(.1,1),(1,3),(3,4.5),(4.5,10),(10,1e2),(1e2,1e4),(1e4,1e6),(1e6,1e9)]:
    k = (y>=lo)&(y<hi); print(f"res y in [{lo},{hi}): {r[k].max().item():.1e}")
s = torch.linspace(1e-6, 1-1e-6, 4001, dtype=C.DT)
T = F.basis3(s, n)[0]; c = p[1:].clone(); hi = c.clone(); hi[:150] = 0
tail = (T @ hi).abs()
ys = ((s ** (-2/al)) - 1).clamp_min(0).sqrt()
for a_, b_ in [(0,.001),(.001,.01),(.01,.05),(.05,.2),(.2,.4),(.4,.6),(.6,.8),(.8,.95),(.95,.99),(.99,.999),(.999,1)]:
    k = (s>=a_)&(s<b_); print(f"high-k content s in [{a_},{b_}) (y ~ {ys[k].max().item():.2g}..{ys[k].min().item():.2g}): {tail[k].max().item():.1e}")
