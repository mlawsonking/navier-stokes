"""1a with origin Taylor rows, batched multi-start at n=48, then continuation in n (pad coefficients
with zeros, polish at n=96 and 192). Initial noise decays like k^-3 so high modes start near zero
(white noise on high modes makes the origin rows, which weigh c_k by ~k^6, enormous)."""
import json, subprocess, time, torch
from gpu_solver import burgers as B
from gpu_solver.batch_lm import batched_lm
dev, DT = "cuda", torch.float64
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() + "+dirty"
ledger = open("runs/ledger.jsonl", "a")
yf = torch.linspace(1e-6, 20, 4001, dtype=DT, device=dev)


def solve(n, u3_zero, P0, meta, stage):
    y = B.collocation_y(2 * n, device=dev)
    D = B.cheb_origin_table(n, 4, dev)
    f = lambda p: B.residual_vec_origin(p, y, n, D, u3_zero=u3_zero)
    t0 = time.time()
    p, sup, it = batched_lm(f, P0, max_iter=300)
    torch.cuda.synchronize(); wall = time.time() - t0
    lam_ref = 0.25 if u3_zero else 0.5
    Ue = B.exact_U(yf, int(round(1 + 1 / lam_ref)))
    npass = 0
    for b in range(p.shape[0]):
        U, Uy, s = B.profile(p[b], yf, n)
        rows, v = B.origin_rows(p[b], D)
        lam = p[b, 0].item()
        rec = dict(case="1a-burgers-origin-cont" + ("-U3zero" if u3_zero else ""), stage=stage, n_basis=n, n_colloc=2 * n,
                   seed=meta[b][0], lam0=meta[b][1], lam=lam, lam_ref=lam_ref, lam_err=abs(lam - lam_ref),
                   profile_sup_err_absy_le_20=(U - Ue).abs().max().item(),
                   residual_sup_absy_le_20=(-p[b, 0] * U + ((1 + p[b, 0]) * yf + U) * Uy).abs().max().item(),
                   residual_sup_all_rows=sup[b].item(), u3=v[1].item(), u5=v[2].item(), u7=v[3].item(),
                   iters=it[b].item(), batch=p.shape[0], wall_s_batch=round(wall, 2), commit=commit,
                   device=torch.cuda.get_device_name(0), date=time.strftime("%Y-%m-%d %H:%M:%S"))
        rec["pass_"] = rec["lam_err"] < 1e-6 and rec["profile_sup_err_absy_le_20"] < 1e-6
        npass += rec["pass_"]
        ledger.write(json.dumps(rec) + "\n")
        print(f"n={n} u3z={u3_zero} seed={meta[b][0]} lam0={meta[b][1]:.3f} lam={lam:.15f} err={rec['lam_err']:.1e} "
              f"prof={rec['profile_sup_err_absy_le_20']:.1e} res={rec['residual_sup_absy_le_20']:.1e} u5={rec['u5']:.6f} it={rec['iters']}", flush=True)
    print(f"== {stage} n={n} u3_zero={u3_zero}: {npass}/{p.shape[0]} pass, batch wall {wall:.1f}s\n", flush=True)
    ledger.flush()
    return p


def chain(u3_zero, lam0s, seeds=(0, 1), n0=48):
    P0, meta = [], []
    k = torch.arange(1, n0 + 1, dtype=DT, device=dev)
    for sd in seeds:
        torch.manual_seed(sd)
        for l0 in lam0s:
            P0.append(torch.cat([torch.tensor([l0], dtype=DT, device=dev), 0.1 * torch.randn(n0, dtype=DT, device=dev) / k ** 3]))
            meta.append((sd, l0))
    p = solve(n0, u3_zero, torch.stack(P0), meta, "multistart")
    for n in (96, 192):
        p = torch.cat([p, torch.zeros(p.shape[0], n - p.shape[1] + 1, dtype=DT, device=dev)], 1)
        p = solve(n, u3_zero, p, meta, "continued")


chain(True, [0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.33, 0.36])
chain(False, [0.40, 0.45, 0.55, 0.62, 0.70, 0.80])
