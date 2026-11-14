import torch, math, sys
from gpu_solver import ccf_sing as S, ccf_fast as F
DEV = "cuda"; x = torch.load("runs/sing_track2_v1.pt")[-1]; p = x["p"].to(DEV); F.MAP = tuple(x["map"]); shape = (640, 8, 4)
G = S.EGrid(h=0.0125, device=DEV, cluster=("box", math.log(x["ypk"]), 0.15, 80.0))
r, J = S.res_jac(p, G, shape, (False, True)); d = J.norm(dim=0)
U, sv, Vh = torch.linalg.svd(J / d, full_matrices=False)
print("count sv < 1e-12:", int((sv < 1e-12 * sv[0]).sum()), " < 1e-8:", int((sv < 1e-8 * sv[0]).sum()), " < 1e-4:", int((sv < 1e-4 * sv[0]).sum()), "of", sv.numel())
names = ["mu"] + [f"N0[{k}]" for k in range(640)] + [f"j{j}[{k}]" for j in range(1, 5) for k in range(8)]
for i in (-1, -5, -20, -40):
    v = Vh[i].abs(); top = v.topk(6)
    blocks = [v[0].item(), v[1:641].norm().item()] + [v[641 + 8 * j:649 + 8 * j].norm().item() for j in range(4)]
    print(f"sv {sv[i].item():.1e}: weight mu {blocks[0]:.2f} N0 {blocks[1]:.2f} j1..j4 {[round(b, 2) for b in blocks[2:]]}; top:", [names[k] for k in top.indices.tolist()])
print("column norms d: N0 first/last", d[1].item(), d[640].item(), " sing blocks", [f"{d[641 + 8 * j].item():.1e}..{d[648 + 8 * j].item():.1e}" for j in range(4)])
