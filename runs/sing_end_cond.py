"""Conditioning of lam at a converged cusp solution: residual cost of a unit change of lam (or mu) when all other
unknowns re-adjust, ||(I - P_rest) J_lam||, for several SVD cutoffs of J_rest; plus the smallest singular values and
the lam / mu / asymmetry content of the corresponding right singular vectors.
Usage: sing_end_cond.py STATE.pt [R_src m]"""
import sys, math, torch
from gpu_solver import ccf_cusp as C
import os
if os.environ.get("RHO"): C.RHO = float(os.environ["RHO"])
st = torch.load(sys.argv[1]); RS = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0; MM = int(sys.argv[3]) if len(sys.argv) > 3 else 2
cu = C.Cusp(st["ystar"], st["nL"], st["nR"], *st["njJ"], nb=st.get("nb", 0), nfar=st.get("nfar", 0)); p = st["p"].to("cuda")
G = C.CuspGrid(cu, p[0].item(), R_src=RS, m=MM)
r, J = C.res_jac(cu, p, G); print(f"rows {J.shape[0]} cols {J.shape[1]}  |r| {r.norm().item():.2e}")
aL, aR = C.amp_rows(cu, p[0].item())
for k, name in ((0, "lam"), (1, "mu")):
    rest = [i for i in range(J.shape[1]) if i != k]; Jr = J[:, rest]; d = Jr.norm(dim=0).clamp_min(1e-300)
    U, sv, Vh = torch.linalg.svd((Jr / d).cpu(), full_matrices=False); jk = J[:, k].cpu()
    for cut in (1e-6, 1e-8, 1e-10, 1e-12, 1e-14):
        kk = sv > cut * sv[0]; Uk = U[:, kk]; res = jk - Uk @ (Uk.T @ jk)
        print(f"  {name}: cutoff {cut:.0e} rank {int(kk.sum())}: ||J_{name}|| {jk.norm().item():.3e}  cost of a unit change {res.norm().item():.3e}")
d = J.norm(dim=0).clamp_min(1e-300); U, sv, Vh = torch.linalg.svd((J / d).cpu(), full_matrices=False)
print("smallest scaled singular values / largest:", " ".join(f"{v:.1e}" for v in (sv[-12:] / sv[0]).tolist()))
for i in range(1, 9):
    v = Vh[-i] / d.cpu(); c = v[2:].to("cuda")
    rho_dir = ((aR @ c) - (aL @ c) * (aR @ p[2:]) / (aL @ p[2:])) / (aL @ p[2:])
    print(f"  sv {sv[-i].item()/sv[0].item():.1e}: dlam {v[0].item():+.2e} dmu {v[1].item():+.2e} d(A_R/A_L) {rho_dir.item():+.2e} |dc_L| {c[:cu.nL].norm().item():.1e} |dc_R| {c[cu.nL:cu.nL+cu.nR].norm().item():.1e} |dc_orig| {c[cu.nL+cu.nR:].norm().item():.1e}")
# residual pattern of the softest lam-carrying direction: unit lam change, rest re-adjusted (cutoff 1e-12)
rest = list(range(1, J.shape[1])); Jr = J[:, rest]; dr = Jr.norm(dim=0).clamp_min(1e-300)
U, sv, Vh = torch.linalg.svd((Jr / dr).cpu(), full_matrices=False); kk = sv > 1e-12 * sv[0]; jk = J[:, 0].cpu()
x = -(Vh[kk].T @ ((U[:, kk].T @ jk) / sv[kk])) / dr.cpu(); res = jk + Jr.cpu() @ x
c = x[1:].to("cuda"); rho_dir = ((aR @ c) - (aL @ c) * (aR @ p[2:]) / (aL @ p[2:])) / (aL @ p[2:])
print(f"soft lam direction: dmu/dlam {x[0].item():+.4e}  d(A_R/A_L)/dlam {rho_dir.item():+.4e}")
nr = G.y.numel(); dd = G.d.cpu(); yy = G.y.cpu(); rr = res[:nr].abs()
for lo, hi, side in ((1e-7, 1e-4, "L"), (1e-4, 1e-2, "L"), (1e-2, 1, "L"), (1, 7, "L"), (1e-7, 1e-4, "R"), (1e-4, 1e-2, "R"), (1e-2, 1, "R"), (1, 1e3, "R"), (1e3, 1e12, "R")):
    m = (dd.abs() >= lo) & (dd.abs() < hi) & ((dd < 0) if side == "L" else (dd > 0))
    if m.any(): print(f"   {side} |d| in [{lo:.0e},{hi:.0e}): rows {int(m.sum())} max|res| {rr[m].max().item():.2e}  rms {rr[m].pow(2).mean().sqrt().item():.2e}")
print(f"   extra rows: {res[nr:].tolist()}")
