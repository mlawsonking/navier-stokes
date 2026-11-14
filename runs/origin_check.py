"""Check the origin series against (i) U'''(0)=3a(1-sum c) and (ii) finite differences of profile()."""
import torch
from gpu_solver import burgers as B
dev, DT = "cuda", torch.float64
torch.manual_seed(0)
n = 12
p = torch.cat([torch.tensor([0.37], dtype=DT, device=dev), 0.3 * torch.randn(n, dtype=DT, device=dev)])
D = B.cheb_origin_table(n, 4, dev)
rows, v = B.origin_rows(p, D)
a = 1 / (1 + p[0])
print("v0", v[0].item(), " u3 series", v[1].item(), " formula a(1-sum c)/2", (a * (1 - p[1:].sum()) / 2).item())
# fit V(t) from profile at small y: (U/y) polynomial in t
y = torch.linspace(0.02, 0.2, 40, dtype=DT, device=dev)
U, Uy, s = B.profile(p, y, n)
A = torch.stack([y ** (2 * j) for j in range(8)], 1)
coef = torch.linalg.lstsq(A, (U / y)[:, None]).solution[:, 0]
print("series v ", v.tolist()); print("fit    v ", coef[:4].tolist())
u3, u5, u7, lam = v[1], v[2], v[3], p[0]
print("rows", rows.tolist())
print("hand", [(u3 * (2 * lam - 1)).item(), (u5 * (4 * lam - 1) + 3 * u3 ** 2).item(), (u7 * (6 * lam - 1) + 8 * u3 * u5).item()])
