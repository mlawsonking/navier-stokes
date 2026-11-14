"""Second-resolution check of the first unstable branch: re-solve selected points with n = 256 and half the
quadrature step (twice the rows), lam free. Prints lam at both resolutions."""
import math, json, torch
from gpu_solver import ccf as C, ccf_fast as F
DEV = "cuda"; n2 = 256
G = C.Grid(h=0.00625, L=40.0, Lc=22.0, device=DEV, n_gl=160)
F.MODE.update(norm="origin", origin_row=True, d=0.1, ymin=0.0)
pts = {}
for f in ("down192b", "up192b", "up192d"):
    for s in torch.load(f"runs/gclm_u1_branch_{f}.pt"):
        pts[round(s["a"], 4)] = s
out = open("runs/gclm_u1_branch.jsonl", "a")
for a in (-1.36, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.74, -0.7):
    s = pts[a]; p = s["p"].to(DEV); mp = tuple(s["map"]); n1 = p.numel() - 1
    j = torch.arange(n2, dtype=C.DT, device=DEV); t = torch.cos(math.pi * (j + 0.5) / n2)
    x0, eps = mp; b1, b2 = math.asinh((1 - x0) / eps), math.asinh((1 + x0) / eps)
    sn = ((x0 + eps * torch.sinh((b1 + b2) * (t + 1) / 2 - b2) + 1) / 2).clamp(1e-300, 1.0)
    F.MAP = mp
    M = F.basis3(sn, n1)[0] @ p[1:]
    q0 = torch.cat([p[:1], torch.linalg.solve(F.basis3(sn, n2)[0], M)])
    q, hist = F.lm(q0, G, a=a, lam_free=True, max_iter=60)
    r = F.residual(q[0], q[1:], G, a)
    print(f"a {a:+.2f} lam(n=192,h=0.0125) {p[0].item():.13f} lam(n=256,h=0.00625) {q[0].item():.13f} diff {abs(p[0].item()-q[0].item()):.1e} sup {r.abs().max().item():.1e} its {len(hist)}", flush=True)
    out.write(json.dumps(dict(case="gclm-u1-branch-verify", a=a, n_basis=n2, quad_h=0.00625, lam=q[0].item(), lam_coarse=p[0].item(), res_sup=r.abs().max().item())) + "\n")
