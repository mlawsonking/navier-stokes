import torch, math
from gpu_solver import ccf as C, ccf_fast as F
torch.set_num_threads(4)
F.MODE.update(norm="origin", origin_row=True, d=0.1, ymin=0.0); F.FAR = None
x = torch.load("runs/gclm_u1_branch_upd256.pt")[0]; p = x['p']; F.MAP = tuple(x['map']); a = x['a']; xp = math.asinh(3.827)
print("a", a, "map", F.MAP)
for name, h, cl in [("fit grid", 0.0125, (xp, 0.5, 0.475)), ("plain h", 0.0125, None), ("plain h/2", 0.00625, None), ("plain h/4", 0.003125, None), ("shift 20x", 0.01, (xp + 0.011, 0.4, 0.38)), ("5x wide", 0.01, (xp, 1.0, 0.8))]:
    G = C.Grid(h=h, L=40.0, Lc=22.0, device="cpu", n_gl=128, cluster=cl)
    r = F.residual(p[0], p[1:], G, a)[:-2].abs(); i = r.argmax(); y = G.y
    srt = r.sort(descending=True)
    print(f"{name:10s} rows {r.numel()} sup {r[i].item():.1e} at y {y[i].item():.4f}; next worst y:", [round(y[j].item(), 3) for j in srt.indices[1:6]], "; count > 1e-10:", int((r > 1e-10).sum()), "y-range of those:", (y[r > 1e-10].min().item() if (r > 1e-10).any() else None), (y[r > 1e-10].max().item() if (r > 1e-10).any() else None))
