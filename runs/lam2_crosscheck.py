"""Cross-check lam_2 (CCF, a = -1) with the second solver gclm_log (representation Theta = r^beta s N + q0 (r^beta-1)/(2 beta),
vorticity-form residual) starting from the ccf_sing smooth solution. Usage: lam2_crosscheck.py src.pt n [h]"""
import sys, json, math, torch
from gpu_solver import ccf_sing as S, ccf_fast as F, gclm_log as L
DEV = "cuda"; A = sys.argv; n = int(A[2]); h = float(A[3]) if len(A) > 3 else 0.0125
x = torch.load(A[1]); p = x["p"].to(DEV); mp = tuple(x["map"]); F.MAP = mp; n0 = p.numel() - 2; lam = p[0]; L.D = S.D
def nodes(n):
    j = torch.arange(n, dtype=S.DT, device=DEV); t = torch.cos(math.pi * (j + 0.5) / n)
    x0, eps = mp; b1, b2 = math.asinh((1 - x0) / eps), math.asinh((1 + x0) / eps)
    return (x0 + eps * torch.sinh((b1 + b2) * (t + 1) / 2 - b2) + 1) / 2
s = nodes(n); zero = torch.zeros(1, dtype=S.DT, device=DEV)
M = lambda s: (S.E2(s, 1 - s, p[1], (n0, 1, 0))[0] @ p[2:])          # Theta = r^beta M(s), M = (1-s) N_0
M0 = M(zero)[0]; beta = lam / (2 * (1 + lam))
q = torch.cat([lam.reshape(1), (2 * beta * M0).reshape(1), torch.linalg.solve(F.basis3(s, n)[0], (M(s) - M0) / s)])
ypk = x.get("ypk", 7.722)
G = S.peak_grid(lam.item(), ypk, mp, n, h=h, R=6, device=DEV)
r0 = L.residual(q[0], q[1:], G, -1.0).abs().max().item()
q, hist = L.gn(q, G, a=-1.0, max_iter=30)
sup = L.residual(q[0], q[1:], G, -1.0).abs().max().item(); del G; torch.cuda.empty_cache()
G2 = S.peak_grid(lam.item(), ypk, mp, n, h=h * 0.8, R=7.8, device=DEV, shift=0.37, widen=1.4)
off = L.residual(q[0], q[1:], G2, -1.0).abs().max().item(); c = q[2:].abs()
print(f"n {n} h {h} start {r0:.1e} -> lam {q[0].item():.14f} (ccf_sing {x['lam']:.14f}, diff {q[0].item()-x['lam']:.1e}) q0 {q[1].item():.10f} sup {sup:.1e} OFF {off:.1e} its {len(hist)} c_last {c[-4:].max().item():.0e}", flush=True)
open("runs/ledger.jsonl", "a").write(json.dumps(dict(case="lam2-crosscheck-gclm_log", src=A[1], n=n, quad_h=h, lam=q[0].item(), lam_src=x["lam"], q0=q[1].item(), res_start=r0, res_sup=sup, res_offgrid=off)) + "\n")
