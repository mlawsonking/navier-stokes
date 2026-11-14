import torch, math
from gpu_solver import ccf as C, ccf_fast as F
F.MODE.update(norm="origin", origin_row=True, d=0.1, ymin=0.0); F.FAR = None
G = C.Grid(h=0.00625, L=40.0, Lc=22.0, device="cuda", n_gl=128)
for f in ["runs/gclm_u1_branch_down192b.pt", "runs/gclm_u1_branch_upw224.pt", "runs/gclm_u1_branch_upc288.pt"]:
    for x in torch.load(f):
        if round(x['a'] * 100) % 4 and x['a'] < -0.56: continue
        p = x['p'].to("cuda"); F.MAP = tuple(x['map']); a = x['a']; lam = p[0]; n = p.numel() - 1
        BOm, BOmy, AU, AUp, up0, q1, wt = F.operators(lam, G, n); c = p[1:]
        U, Up, Om, Omy = AU @ c, AUp @ c, BOm @ c, BOmy @ c
        V = (1 + lam) * G.y + a * U; rel = V / ((1 + lam) * G.y); i = rel.argmin()
        R = ((Om + V * Omy - Om * Up) / wt).abs()
        j = Om.abs().argmax()
        print(f"{f[-12:-3]:9s} a {a:+.2f} lam {lam.item():.6f} min V/((1+lam)y) {rel[i].item():.5f} at y {G.y[i].item():.3f}; U'-1 there {Up[i].item()-1:+.4f}; peak y {G.y[j].item():.3f}; res sup (h/2 plain grid) {R.max().item():.1e}")
