import torch, time, math
from gpu_solver import ccf_sing as S, ccf_fast as F
DEV = "cuda"; x = torch.load("runs/ccf_u2_a.pt"); p = x["p"].to(DEV); F.MAP = tuple(x["map"]); shape = (640, 8, 4)
G = S.EGrid(h=0.00625, device=DEV, cluster=("box", math.log(7.722), 0.15, 40.0))
def T(f, k=3):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(k): f()
    torch.cuda.synchronize(); return round((time.time() - t) / k, 3)
print("rows", G.y.numel(), "sources", G.ys.numel())
print("residual", T(lambda: S.residual(p, G, shape)), "res_jac", T(lambda: S.res_jac(p, G, shape, (False, True))))
print("fields(ys)", T(lambda: S._fields(p[0], p[1], G.ys, shape)), "basis3 only", T(lambda: F.basis3(G.ys.clamp(max=1.0), 640)))
B = torch.randn(G.ys.numel(), 672, dtype=S.DT, device=DEV)
print("K@B", T(lambda: G.K @ B))
r, J = S.res_jac(p, G, shape, (False, True))
m = J.shape[1]; A = torch.cat([J, torch.eye(m, dtype=S.DT, device=DEV)])
print("QR gpu", T(lambda: torch.linalg.qr(A)), "QR cpu", T(lambda: torch.linalg.qr(A.cpu())), "chol normal eq", T(lambda: torch.linalg.cholesky(A.T @ A)))
