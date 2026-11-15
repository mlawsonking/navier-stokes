import torch, math
from gpu_solver import ccf as C, ccf_fast as F, gclm_log as L, ccf_sing as S
torch.set_num_threads(6); L.D = 0.1
x = torch.load("runs/gclm_log_branch_w8.pt")[-1]; p = x["p"]; F.MAP = tuple(x["map"]); a = x["a"]; n = p.numel() - 2; yp = x["ypk"]
print("a", a, "lam", p[0].item(), "map", F.MAP, "ypk", yp, "n", n)
G = S.peak_grid(p[0].item(), yp, F.MAP, n, h=0.0125, R=8, device="cpu")
r = L.residual(p[0], p[1:], G, a); R = r[:-2].abs(); y = G.y; print("rows", y.numel(), "extra rows", r[-2:].tolist())
for lo, hi in [(0, 1e-3), (1e-3, .03), (.03, .2), (.2, .5), (.5, .59), (.59, .61), (.61, .7), (.7, 1.5), (1.5, 10), (10, 1e3), (1e3, 1e10)]:
    k = (y >= lo) & (y < hi); print(f"y [{lo},{hi}): rows {int(k.sum())} max res {R[k].max().item():.1e}")
c = p[2:].abs(); print("coef blocks:", " ".join(f"{c[k:k+n//16].max().item():.0e}" for k in range(0, n, n // 16)))
