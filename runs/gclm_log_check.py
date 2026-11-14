"""gclm_log sanity: (1) exact a = 1/2 solution (lam = -2/3); (2) converted branch solutions keep their residual."""
import torch, math
from gpu_solver import ccf as C, ccf_fast as F, gclm_log as L
torch.set_num_threads(8)
def nodes(n, mp):
    j = torch.arange(n, dtype=C.DT); t = torch.cos(math.pi * (j + 0.5) / n)
    if mp is None: return (t + 1) / 2
    x0, eps = mp; b1, b2 = math.asinh((1 - x0) / eps), math.asinh((1 + x0) / eps)
    return (x0 + eps * torch.sinh((b1 + b2) * (t + 1) / 2 - b2) + 1) / 2
# (1) exact: Omega = -(16/3) y/(1+y^2)^2, lam = -2/3: alpha = 3, beta = -1, s = r^(-3/2).
# Omega = y r^(beta-1) Qtot = y r^-2 Qtot -> Qtot = -16/3 (constant) = q0 + Q_N with N = 0.  d = 16/3.
F.MAP = None; L.D = 16 / 3
G = C.Grid(h=0.0125, L=40.0, Lc=22.0, device="cpu", n_gl=64)
p = torch.zeros(2 + 12, dtype=C.DT); p[0] = -2 / 3; p[1] = -16 / 3
r = L.residual(p[0], p[1:], G, 0.5); print("exact a=1/2: residual sup", r.abs().max().item())
g = p.clone(); g[0] += 0.03; g[1] *= 1.05; g[2:] += 0.01 * torch.randn(12, dtype=C.DT)
q, h = L.gn(g, G, a=0.5); print("  GN from a perturbed start: lam", q[0].item(), "error", abs(q[0].item() + 2 / 3), "its", len(h), "sup", h[-1][1])
# (2) conversion
L.D = 0.1
for x in torch.load("runs/gclm_u1_branch_upl256.pt")[::4]:
    F.MAP = tuple(x["map"]); a = x["a"]; po = x["p"]; n = po.numel() - 1
    G = C.Grid(h=0.0125, L=40.0, Lc=22.0, device="cpu", n_gl=128, cluster=(math.asinh(3.44), 0.5, 0.475))
    F.MODE.update(norm="origin", origin_row=True, d=0.1, ymin=0.0)
    r_old = F.residual(po[0], po[1:], G, a).abs().max().item()
    pn = L.from_fast(po, n, nodes(n, F.MAP)); r_new = L.residual(pn[0], pn[1:], G, a).abs().max().item()
    q, h = L.gn(pn, G, a=a)
    print(f"a {a:+.3f} lam {po[0].item():.12f}: residual old rep {r_old:.1e}, converted {r_new:.1e}; after GN lam {q[0].item():.12f} sup {h[-1][1] if h else float('nan'):.1e} its {len(h)} q0 {q[1].item():.6f}")
