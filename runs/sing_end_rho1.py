"""Secant iteration in lam (each step a FIXED-lam solve by sing_end_solve.py) for the member of the singular cusp
family with A_R/A_L = TARGET (default 1, the even inner core). Usage:
    sing_end_rho1.py tag nL nR lam_a lam_b INIT.pt [R_src m]      env NB, NFAR, YST, TARGET passed through"""
import sys, os, json, subprocess
A = sys.argv; tag, nL, nR, la, lb, init = A[1], A[2], A[3], float(A[4]), float(A[5]), A[6]; extra = A[7:]
T = float(os.environ.get("TARGET", 1.0)); py = os.path.abspath(".venv/Scripts/python.exe")
def run(lam, start, name):
    env = dict(os.environ, LAMFIX=repr(lam), INIT=start, ITERS=os.environ.get("ITERS", "25"))
    out = subprocess.run([py, "-u", "runs/sing_end_solve.py", name, nL, nR] + extra, env=env, capture_output=True, text=True).stdout
    open(f"runs/sing_end_{name}.log", "w").write(out)
    row = [json.loads(l) for l in open("runs/sing_end.jsonl") if json.loads(l)["case"] == "sing-end-" + name][-1]
    print(f"  lam {lam:.12f}: rho {row['rho']:.12f} mu {row['mu']:.12f} y*_std {row['y_star_std']:.10f} sup {row['res_sup']:.1e} off {row['res_offgrid']:.1e}", flush=True)
    return row
ra = run(la, init, f"{tag}_a"); rb = run(lb, f"runs/sing_end_{tag}_a.pt", f"{tag}_b"); pts = [(la, ra), (lb, rb)]
for it in range(6):
    (l0, r0), (l1, r1) = pts[-2], pts[-1]
    l2 = l1 + (T - r1["rho"]) * (l1 - l0) / (r1["rho"] - r0["rho"])
    r2 = run(l2, f"runs/sing_end_{tag}_{'b' if it == 0 else it - 1}.pt", f"{tag}_{it}"); pts.append((l2, r2))
    if abs(r2["rho"] - T) < 2e-11: break
l, r = pts[-1]
print(f"FINAL {tag}: A_R/A_L = {r['rho']:.12f} at lam {l:.12f} mu {r['mu']:.12f} U'(0) {r['up0']:.12f} y*_std {r['y_star_std']:.10f}  sup {r['res_sup']:.1e} off {r['res_offgrid']:.1e}"
      f"  dlam/drho {(pts[-1][0]-pts[-2][0])/(pts[-1][1]['rho']-pts[-2][1]['rho']):.5f}")
