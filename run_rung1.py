"""Rung 1 driver: 1a Burgers (lam free; lam free + U'''(0)=0) and 1b CLM.

Appends one JSON line per case to runs/ledger.jsonl.
Usage: .venv/Scripts/python run_rung1.py
"""
import json, math, subprocess, time
import torch
from gpu_solver.lm import levenberg_marquardt
from gpu_solver import burgers as B, clm as C

dev = "cuda"
DT = torch.float64
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = bool(subprocess.run(["git", "status", "--porcelain", "gpu_solver", "run_rung1.py"], capture_output=True, text=True).stdout.strip())
ledger = open("runs/ledger.jsonl", "a")


def log(rec):
    rec.update(commit=commit + ("+dirty" if dirty else ""), device=torch.cuda.get_device_name(0),
               date=time.strftime("%Y-%m-%d %H:%M:%S"))
    ledger.write(json.dumps(rec) + "\n"); ledger.flush()
    print(json.dumps(rec))


def burgers_case(n, lam0, u3_zero, seed):
    torch.manual_seed(seed)
    y = B.collocation_y(2 * n, device=dev)
    p0 = torch.cat([torch.tensor([lam0], dtype=DT, device=dev), 1e-2 * torch.randn(n, dtype=DT, device=dev)])
    t0 = time.time()
    p, hist = levenberg_marquardt(lambda q: B.residual_vec(q, y, n, u3_zero), p0, max_iter=1000)
    torch.cuda.synchronize(); wall = time.time() - t0
    lam = p[0].item()
    lam_ref = 0.25 if u3_zero else 0.5
    m = int(round(1 + 1 / lam_ref))
    yf = torch.linspace(1e-6, 20, 4001, dtype=DT, device=dev)
    U, Uy, s = B.profile(p, yf, n)
    Ue = B.exact_U(yf, m)
    prof_err = (U - Ue).abs().max().item()
    res_raw = (-p[0] * U + ((1 + p[0]) * yf + U) * Uy).abs().max().item()   # unnormalised, |y|<=20
    res_col = B.residual_vec(p, y, n, u3_zero).abs().max().item()
    # which exact branch lam = 1/(2i) did it land on, and does it match that branch's profile?
    i_br = max(1, round(1 / (2 * lam)))
    lam_br = 1 / (2 * i_br)
    br_err = (U - B.exact_U(yf, 2 * i_br + 1)).abs().max().item()
    ok = abs(lam - lam_ref) < 1e-6 and prof_err < 1e-6
    return dict(case="1a-burgers" + ("-U3zero" if u3_zero else ""), n_basis=n, n_colloc=2 * n, seed=seed,
                lam0=lam0, lam=lam, lam_ref=lam_ref, lam_err=abs(lam - lam_ref),
                profile_sup_err_absy_le_20=prof_err, residual_sup_absy_le_20=res_raw,
                residual_sup_colloc_normalised=res_col, iters=len(hist), wall_s=round(wall, 3),
                nearest_branch_lam=lam_br, nearest_branch_lam_err=abs(lam - lam_br),
                nearest_branch_profile_err=br_err, pass_=ok)


def clm_case(K, c_free, c0, seed):
    torch.manual_seed(seed)
    th = C.colloc_theta(2 * K, device=dev)
    b0 = torch.zeros(K, dtype=DT, device=dev); b0[0] = -0.7
    b0[1:] = 0.05 * torch.randn(K - 1, dtype=DT, device=dev) / torch.arange(2, K + 1, device=dev) ** 2
    p0 = torch.cat([torch.tensor([c0], dtype=DT, device=dev), b0]) if c_free else b0
    t0 = time.time()
    f = (lambda q: C.residual_vec(q, th)) if c_free else (lambda q: C.residual_vec(q, th, c_fixed=1.0))
    p, hist = levenberg_marquardt(f, p0, max_iter=300)
    torch.cuda.synchronize(); wall = time.time() - t0
    c = p[0].item() if c_free else 1.0
    b = p[1:] if c_free else p
    yf = torch.linspace(-50, 50, 20001, dtype=DT, device=dev)
    thf = 2 * torch.atan(yf)
    F, yWy, HW = C.fields(b, thf)
    We = -2 * yf / (1 + yf ** 2)
    prof_err = (F - We).abs().max().item()
    res = (F + c * yWy - F * HW).abs().max().item()
    ok = res < 1e-8 and prof_err < 1e-6 and abs(c - 1) < 1e-8
    return dict(case="1b-clm" + ("-c_free" if c_free else "-c_fixed1"), K_modes=K, n_colloc=2 * K, seed=seed,
                c0=c0 if c_free else None, c=c, residual_sup_absy_le_50=res, profile_sup_err_absy_le_50=prof_err,
                b1=b[0].item(), max_abs_bk_k_ge_2=b[1:].abs().max().item(), iters=len(hist), wall_s=round(wall, 3),
                pass_=ok)


if __name__ == "__main__":
    import sys
    torch.zeros(1, device=dev)  # warm up CUDA
    if len(sys.argv) > 1 and sys.argv[1] == "refine":
        # basis refinement for the convergence-rate check (see HANDOFF.md)
        for n in (96, 192):
            for seed in (0, 1):
                log(burgers_case(n, 0.55, False, seed))
                log(burgers_case(n, 0.40, False, seed))
                log(burgers_case(n, 0.30, True, seed))
        sys.exit()
    for n in (24, 48):
        for seed in (0, 1):
            for lam0 in (0.40, 0.45, 0.55, 0.70):
                log(burgers_case(n, lam0, False, seed))
            for lam0 in (0.20, 0.30):
                log(burgers_case(n, lam0, True, seed))
    for K in (16, 32):
        for seed in (0, 1):
            log(clm_case(K, False, None, seed))
            for c0 in (0.8, 1.3):
                log(clm_case(K, True, c0, seed))
