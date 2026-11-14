"""Which discretisation parameter limits the first unstable branch at a=-0.7?"""
import sys, torch, math, time
from gpu_solver import ccf as C, ccf_fast as F
sols = torch.load("runs/gclm_u1_branch_up192d.pt")
F.MODE.update(norm="origin", origin_row=True, d=0.1, ymin=0.0)
s0 = [s for s in sols if abs(s['a'] + 0.7) < 1e-9][0]
a = s0['a']; DEV = "cuda"
def remap(p, old, new, n):
    j = torch.arange(n, dtype=C.DT, device=DEV); t = torch.cos(math.pi * (j + 0.5) / n)
    if new is None: s = ((t + 1) / 2)
    else:
        x0, eps = new; b1, b2 = math.asinh((1 - x0) / eps), math.asinh((1 + x0) / eps)
        s = ((x0 + eps * torch.sinh((b1 + b2) * (t + 1) / 2 - b2) + 1) / 2).clamp(1e-300, 1.0)
    F.MAP = old; M = F.basis3(s, p.numel() - 1)[0] @ p[1:]
    F.MAP = new; return torch.cat([p[:1], torch.linalg.solve(F.basis3(s, n)[0], M)])
def run(tag, n=192, h=0.0125, L=40.0, Lc=22.0, ngl=128, mp=tuple(s0['map'])):
    t0 = time.time()
    G = C.Grid(h=h, L=L, Lc=Lc, device=DEV, n_gl=ngl)
    p = remap(s0['p'].to(DEV), tuple(s0['map']), mp, n)
    q, hist = F.lm(p, G, a=a, lam_free=True, max_iter=60)
    r = F.residual(q[0], q[1:], G, a); c = q[1:].abs()
    print(f"{tag:28s} lam {q[0].item():.12f} sup {r.abs().max().item():.1e} its {len(hist)} coef:", " ".join(f"{c[i:i+n//8].max().item():.0e}" for i in range(0, n, n//8)), f"t={time.time()-t0:.0f}s", flush=True)
x0, eps = s0['map']
run("base")
run("ngl 256", ngl=256)
run("L 60", L=60.0)
run("Lc 30", Lc=30.0)
run("eps x2", mp=(x0, 2 * eps))
run("eps x4", mp=(x0, 4 * eps))
run("n 288", n=288, h=0.00625)
run("n 288 eps x2", n=288, h=0.00625, mp=(x0, 2 * eps))
