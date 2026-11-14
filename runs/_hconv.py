import torch, math
from gpu_solver import ccf as C, ccf_fast as F
F.MODE.update(norm="origin", origin_row=True, d=0.1, ymin=0.0); F.FAR = None
x = [t for t in torch.load("runs/gclm_u1_branch_upw224.pt") if abs(t['a'] + 0.60) < 1e-9][0]
p = x['p'].to("cuda"); F.MAP = tuple(x['map']); a = x['a']; lam = p[0]; n = p.numel() - 1; c = p[1:]
def ev(h, ngl=128):
    G = C.Grid(h=h, L=40.0, Lc=22.0, device="cuda", n_gl=ngl)
    BOm, BOmy, AU, AUp, up0, q1, wt = F.operators(lam, G, n)
    U, Up, Om, Omy = AU @ c, AUp @ c, BOm @ c, BOmy @ c
    R = (Om + ((1 + lam) * G.y + a * U) * Omy - Om * Up) / wt
    return G.y, U, Up, R, (up0 @ c).item()
y1, U1, Up1, R1, u01 = ev(0.0125)
y3, U3, Up3, R3, u03 = ev(0.0125 / 3)
i3 = 3 * torch.arange(y1.numel(), device="cuda") + 1
print("grid points match", (y3[i3] - y1).abs().max().item(), " U'(0):", u01, u03)
dU, dUp = (U3[i3] - U1).abs(), (Up3[i3] - Up1).abs()
for lo, hi in [(0,.1),(.1,1),(1,3),(3,3.5),(3.5,4),(4,5),(5,10),(10,1e3),(1e3,1e9)]:
    k = (y1>=lo)&(y1<hi); k3 = (y3>=lo)&(y3<hi)
    print(f"y [{lo},{hi}): |dU| {dU[k].max().item():.1e} |dU'| {dUp[k].max().item():.1e}  res(h) {R1[k].abs().max().item():.1e} res(h/3) {R3[k3].abs().max().item():.1e} res(h/3 at coarse pts) {R3[i3][k].abs().max().item():.1e}")
