"""Re-solve a profile in the sinh-mapped basis. The start is a least-squares fit of the mapped basis to Omega of
an existing solution (plain basis), then LM with lam free.
Usage: ccf_mapped.py src.pt out.pt n h d x0 eps max_iter [src_map_x0 src_map_eps]"""
import sys, os, time, torch
from gpu_solver import ccf as C, ccf_fast as F
src, outp, n, h, d, x0, eps, mi = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), int(sys.argv[8])
DEV = os.environ.get("DEV", "cpu")
F.MODE.update(norm='origin', origin_row=True, d=d)
G = C.Grid(h=h, L=40.0, Lc=22.0, device=DEV, n_gl=int(os.environ.get("NGL", 96)))
g = torch.load(src).to(DEV).reshape(-1)
F.MAP = (float(sys.argv[9]), float(sys.argv[10])) if len(sys.argv) > 10 else None
Og = F.operators(g[0], G, g.numel() - 1)[0] @ g[1:]
F.MAP = (x0, eps)
B = F.operators(g[0], G, n)[0]
wts = 1 / (G.y * (1 + G.y ** 2) ** (-(1 + 1 / (1 + g[0].item())) / 2))
A = (B * wts[:, None]).cpu(); rhs = (Og * wts).cpu()
c = torch.linalg.lstsq(A, rhs[:, None], driver="gelsd").solution[:, 0].to(DEV)
print(f"fit of mapped basis (n={n}, x0={x0}, eps={eps}) to source profile: weighted sup err {((B @ c - Og) * wts).abs().max().item():.2e}", flush=True)
p = torch.cat([g[:1], c]); t0 = time.time()
p, hist = F.lm(p, G, lam_free=True, max_iter=mi)
r = F.residual(p[0], p[1:], G); ops = F.operators(p[0], G, n); Om = ops[0] @ p[1:]; i = Om.abs().argmax(); cc = p[1:].abs()
print(f"n={n} h={h} map=({x0},{eps}): lam {p[0].item():.14f} sup {r.abs().max().item():.2e} its {len(hist)} U'(0) {(ops[4] @ p[1:]).item():.12f} "
      f"peak {Om[i].abs().item():.6f}@{G.y[i].item():.4f} |c| " + " ".join(f"{cc[j].item():.0e}" for j in range(0, n, max(1, n // 8))) + f" last {cc[-3:].max().item():.1e}  {time.time()-t0:.0f}s", flush=True)
torch.save(p, outp)
