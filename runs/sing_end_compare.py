"""Compare the one-sided cusp amplitudes A_L(d), A_R(d) = -Omega sqrt|d| of cusp solutions (sing_end_*.pt) with the
last family state, at the same |d| (both in the family's internal scale when y* = family y*).
Usage: sing_end_compare.py STATE.pt [STATE2.pt ...]"""
import sys, math, torch
from gpu_solver import ccf_cusp as C, ccf_sing as S, ccf_fast as F
x = torch.load("runs/sing_track4_v4.pt")[-1]; nj, J = x["njJ"]; pf = x["p"].to("cuda"); nf = pf.numel() - 2 - nj * J
ds = [1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 2.0]
def fam(d):
    F.MAP = tuple(x["map"]); S.CENTER = x["center"]; y = torch.tensor([x["ypk"] - d, x["ypk"] + d], dtype=S.DT, device="cuda")
    om = S._fields(pf[0], pf[1], y, (nf, nj, J))[1] @ pf[2:]; return (-om * math.sqrt(d)).tolist()
print("family peak %.1f: " % x["peak"] + "  ".join(f"{d:.0e}:{(lambda a: a[1]/a[0])(fam(d)):.6f}" for d in ds))
for f in sys.argv[1:]:
    st = torch.load(f); cu = C.Cusp(st["ystar"], st["nL"], st["nR"], *st["njJ"], nb=st.get("nb", 0), nfar=st.get("nfar", 0)); p = st["p"].to("cuda")
    out = []
    for d in ds:
        dd = torch.tensor([-d, d], dtype=S.DT, device="cuda"); om = cu.fields(p[0].item(), p[1].item(), st["ystar"] + dd, dd)[1] @ p[2:]
        out.append((om[1] / om[0]).item())
    print(f"{f} lam {st['lam']:.9f}: " + "  ".join(f"{d:.0e}:{v:.6f}" for d, v in zip(ds, out)))
