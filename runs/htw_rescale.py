"""Dynamic rescaling of gCLM from ODD DEGENERATE data (Huang-Tong-Wang, arXiv:2603.25104, odd case):
    Om_tau + (c_l X + a U) Om_X = (c_w + U_X) Om,  U_X = H Om,  U(0) = 0
c_w, c_l chosen each instant so that F1 = H Om(0) = 2 and F2 = (H Om)_XX(0) = 4 stay fixed. gamma = -c_l/c_w = 1 + lam.
Checks: a = 0 must give c_l = 1/2, c_w = -1 for all tau (their Thm 2.4); a = -0.1 should give gamma = 0.7827 (Table 5.1).
Usage: htw_rescale.py a N eps tau_end [tag]   env: A (cluster strength, 2), CFL (0.4), K (vanishing order, 3)"""
import sys, os, json, math, time, torch
DT = torch.float64; DEV = "cuda"; A_ = sys.argv
a, N, eps, tau_end = float(A_[1]), int(A_[2]), float(A_[3]), float(A_[4]); tag = A_[5] if len(A_) > 5 else "t"
CA = float(os.environ.get("A", 2)); CFL = float(os.environ.get("CFL", 0.4)); KV = int(os.environ.get("K", 3)); XMAX = 1e7
xi_of = lambda X: torch.asinh(X) + CA * (torch.asinh((X - 1) / eps) + math.asinh(1 / eps))
ximax = xi_of(torch.tensor(XMAX, dtype=DT)).item(); h = ximax / N
xi = h * torch.arange(N + 1, dtype=DT, device=DEV)
lo = torch.zeros_like(xi); hi = torch.full_like(xi, XMAX * 1.0001)
for _ in range(200):
    mid = (lo + hi) / 2; m = xi_of(mid) < xi; lo = torch.where(m, mid, lo); hi = torch.where(m, hi, mid)
X = (lo + hi) / 2; X[0] = 0.0
gp = 1 / (1 / torch.sqrt(1 + X * X) + CA / torch.sqrt(eps * eps + (X - 1) ** 2))      # dX/dxi
i = torch.arange(N + 1, device=DEV); par = ((i[:, None] - i[None, :]) % 2 == 1).to(DT)
Km = (2 / math.pi) * 2 * h * par * (X * gp)[None, :] / (X[:, None] ** 2 - X[None, :] ** 2 + (1 - par))   # H for odd f, alternate-point PV rule
del par
w = h * gp.clone(); w[0] *= 0.5; w[-1] *= 0.5                                         # trapezoid weights in X
Xs = X.clone(); Xs[0] = 1.0
def over(f, p):                    # f / X^p with the X = 0 value copied from the first node (f ~ X^k, k >= p)
    q = f / Xs ** p; q[0] = q[1] if p >= KV else 0.0; return q
def ddx(f, V):                     # 3rd-order upwind-biased d/dxi, divided by dX/dxi; odd extension at 0, zero beyond the end
    fe = torch.cat([-f[1:3].flip(0), f, torch.zeros(2, dtype=DT, device=DEV)])
    fm2, fm1, f0, fp1, fp2 = fe[:-4], fe[1:-3], fe[2:-2], fe[3:-1], fe[4:]
    dpos = (fm2 - 6 * fm1 + 3 * f0 + 2 * fp1) / (6 * h); dneg = (-fp2 + 6 * fp1 - 3 * f0 - 2 * fm1) / (6 * h)
    return torch.where(V >= 0, dpos, dneg) / gp
def fields(Om):
    Ux = Km @ Om
    inc = 0.5 * h * (Ux[1:] * gp[1:] + Ux[:-1] * gp[:-1]); U = torch.cat([torch.zeros(1, dtype=DT, device=DEV), torch.cumsum(inc, 0)])
    return Ux, U
F1 = lambda f: -(2 / math.pi) * (w @ over(f, 1)); F2 = lambda f: -(4 / math.pi) * (w @ over(f, 3))
def rhs(Om):
    Ux, U = fields(Om)
    Nl = Ux * Om - a * U * ddx(Om, a * U)                      # nonlinear part; its own advection upwinded by a U
    f1, f2 = F1(Om), F2(Om)
    cw = -F1(Nl) / f1; cl = (cw * f2 + F2(Nl)) / (2 * f2)
    V = cl * X + a * U
    return -V * ddx(Om, V) + (cw + Ux) * Om, cl.item(), cw.item(), V
f = -X ** KV / (1 + X * X) ** ((KV + 3) / 2)
ell = math.sqrt((F2(f) / F1(f)).item() / 2.0); f = -(X / ell) ** KV / (1 + (X / ell) ** 2) ** ((KV + 3) / 2); Om = f * (2.0 / F1(f).item())
print(f"a {a} N {N} eps {eps} h {h:.4f} min dX {(h*gp).min().item():.1e} F1 {F1(Om).item():.6f} F2 {F2(Om).item():.6f}", flush=True)
tau, nxt, t0, step = 0.0, 0.0, time.time(), 0; out = open("runs/htw_rescale.jsonl", "a")
while tau < tau_end + 1e-9:
    k1, cl, cw, V = rhs(Om)
    if tau >= nxt - 1e-12:
        j = Om.abs().argmax(); hw = (Om.abs() > Om.abs()[j] / 2).nonzero().flatten(); fw = (X[hw[-1]] - X[hw[0]]).item()
        print(f"tau {tau:7.3f} c_l {cl:.6f} c_w {cw:.6f} gamma {-cl/cw:.6f} peak {Om.abs()[j].item():.3e} at X {X[j].item():.5f} fwhm {fw:.2e} ({len(hw)} pts) F1 {F1(Om).item():.6f} F2 {F2(Om).item():.6f} steps {step} t={time.time()-t0:.0f}s", flush=True)
        out.write(json.dumps(dict(case="htw-" + tag, a=a, N=N, eps=eps, tau=tau, c_l=cl, c_w=cw, gamma=-cl / cw, peak=Om.abs()[j].item(), x_peak=X[j].item(), fwhm=fw, n_fwhm=len(hw))) + "\n"); out.flush()
        nxt += 0.25
        if not math.isfinite(cl): break
    dt = min(CFL * h / (V.abs() / gp).max().item(), nxt - tau if nxt > tau else 1e9, 0.01)
    k2 = rhs(Om + 0.5 * dt * k1)[0]; k3 = rhs(Om + 0.5 * dt * k2)[0]; k4 = rhs(Om + dt * k3)[0]
    Om = Om + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4); tau += dt; step += 1
