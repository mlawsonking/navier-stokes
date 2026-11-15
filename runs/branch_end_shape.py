"""Shape of the first unstable profile as the branch approaches its end: peak height, position, width,
U' extremes. Tests whether the peak is a simple pole approaching the real axis (peak * FWHM -> const)."""
import torch, math, sys
from gpu_solver import ccf as C, ccf_fast as F, gclm_log as L
torch.set_num_threads(4)
rows = []
for f in sys.argv[1:]:
    for x in torch.load(f):
        if "ypk" not in x: continue
        rows.append(x)
seen = set()
for x in sorted(rows, key=lambda t: t["a"]):
    a = x["a"]
    if round(a, 3) in seen or (a < -0.2 and round(a * 1000) % 40): continue
    seen.add(round(a, 3))
    p = x["p"]; F.MAP = tuple(x["map"]); n = p.numel() - 2; yp = x["ypk"]
    y = yp * torch.exp(torch.linspace(-1.5, 1.5, 60001, dtype=C.DT))
    B = L._basis(p[0], y, n); Om = B[1] @ p[1:]
    i = Om.abs().argmax(); pk = Om[i].abs().item(); hw = (Om.abs() > pk / 2).nonzero().flatten()
    fw = (y[hw[-1]] - y[hw[0]]).item()
    lam = p[0].item(); u0 = (lam + 2) / (1 - a)
    print(f"a {a:+.5f} lam {lam:+.6f} lam+a {lam+a:+.5f} U'(0) {u0:.5f} q0 {p[1].item():+.5f} peak {pk:9.3f} y* {y[i].item():.4f} FWHM {fw:.5f} FWHM/y* {fw/y[i].item():.5f} peak*FWHM/y* {pk*fw/y[i].item():.4f} peak*y*^k: k=1 {pk*y[i].item():.2f}")
