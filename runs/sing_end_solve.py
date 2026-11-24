"""Direct solve for the singular end profile of the non-smooth CCF family (gpu_solver/ccf_cusp.py).
Usage: sing_end_solve.py tag nL nR [R_src m]   (m rows per source interval)
env: SRC  family state file (default runs/sing_track4_v4.pt, last state) used for the first guess,
     INIT a previous sing_end_*.pt to start from (re-fitted onto the new basis), DEX core exclusion |d| for the fit,
     H0 far spacing in u. Writes runs/sing_end_<tag>.pt and a line in runs/sing_end.jsonl."""
import sys, os, json, math, time, torch
from gpu_solver import ccf_cusp as C, ccf_sing as S, ccf_fast as F
A = sys.argv; tag, nL, nR = A[1], int(A[2]), int(A[3])
RS = float(A[4]) if len(A) > 4 else 4.0; MM = int(A[5]) if len(A) > 5 else 2
H0 = float(os.environ.get("H0", 0.05)); DEX = float(os.environ.get("DEX", 0.0 if os.environ.get("INIT") else 1e-3)); DEV = "cuda"; t0 = time.time()
src = os.environ.get("SRC", "runs/sing_track4_v4.pt"); x = torch.load(src)[-1]; nj, J = x["njJ"]
nj, J = int(os.environ.get("NJ", nj)), int(os.environ.get("JJ", J))
init = os.environ.get("INIT")
if init:
    st = torch.load(init); yst = st["ystar"]; lam0 = st["p"][0].item()
else:
    yst = float(os.environ.get("YST", x["ypk"])); lam0 = x["p"][0].item()      # YST: another scale (Theta -> a Theta(y/a))
NB, NFAR = int(os.environ.get("NB", 0)), int(os.environ.get("NFAR", 0))
cu = C.Cusp(yst, nL, nR, nj, J, nb=NB, nfar=NFAR, kf=float(os.environ.get("KF", C.KF)))
if os.environ.get("RHO"): C.RHO = float(os.environ["RHO"]); print(f"imposed A_R/A_L = {C.RHO}")
G = C.CuspGrid(cu, lam0, R_src=RS, m=MM, h0=H0)
print(f"y* {yst:.9f} nL {nL} nR {nR} origin blocks {nj}x{J}; sources {G.ys.numel()} rows {G.y.numel()} Y {G.Y.item():.2e} "
      f"rows y {G.y.min().item():.1e}..{G.y.max().item():.1e} |d|min {G.d.abs().min().item():.1e}", flush=True)
ratio = ((G.d[:, None] - G.ds[None, :]).abs() / G.w[None, :]).min().item(); print(f"  min |y_row - y_src| / w = {ratio:.3f}")

# first guess: least-squares fit of a known profile at points away from the core
def fit(lam, mu, yv, dv, th):
    BTh, _, oms, rb, _ = cu.fields(lam, mu, yv, dv); wt = oms ** (mu / 2) * rb
    c = torch.linalg.lstsq((BTh / wt[:, None]).cpu(), (th / wt).cpu()[:, None]).solution[:, 0].to(DEV)
    return c, ((BTh @ c - th) / wt).abs().max().item()
yv = torch.cat([G.ys, G.y]); dv = torch.cat([G.ds, G.d]); m = (dv.abs() > DEX) & (yv < 1e4) & (yv > 1e-6); yv, dv = yv[m], dv[m]
if init:
    cu0 = C.Cusp(yst, st["nL"], st["nR"], *st["njJ"], nb=st.get("nb", 0), nfar=st.get("nfar", 0)); p0 = st["p"].to(DEV)
    th = cu0.fields(p0[0].item(), p0[1].item(), yv, dv)[0] @ p0[2:]; lam_, mu_ = p0[0].item(), p0[1].item()
else:
    pf = x["p"].to(DEV); F.MAP = tuple(x["map"]); S.CENTER = x["center"]; nf = pf.numel() - 2 - nj * J
    asc = yst / x["ypk"]
    th = asc * torch.cat([S._fields(pf[0], pf[1], yy / asc, (nf, nj, J))[0] @ pf[2:] for yy in yv.split(4096)]); lam_, mu_ = pf[0].item(), pf[1].item()
if init and tuple(st["njJ"]) == (nj, J) and st["nL"] <= nL and st["nR"] <= nR and abs(st["ystar"] - yst) == 0         and st.get("nb", 0) <= NB and st.get("nfar", 0) <= NFAR:
    # same family of blocks: pad every Chebyshev series with zeros (exact; a least-squares refit loses the sloppy directions)
    q = st["p"].to(DEV)[2:]; z = lambda k: torch.zeros(k, dtype=S.DT, device=DEV); a0, b0, nb0, nf0 = st["nL"], st["nR"], st.get("nb", 0), st.get("nfar", 0)
    o0 = a0 + b0 + nj * J
    c = torch.cat([q[:a0], z(nL - a0), q[a0:a0 + b0], z(nR - b0), q[a0 + b0:o0],
                   q[o0:o0 + nb0], z(NB - nb0), q[o0 + nb0:o0 + 2 * nb0], z(NB - nb0), q[o0 + 2 * nb0:], z(NFAR - nf0)]); err = 0.0
elif not init:
    # family start: fit with a short series (the long one is unconstrained inside the excluded core) and pad
    nfit = int(os.environ.get("FITN", 64)); cu_s = cu; cu = C.Cusp(yst, min(nfit, nL), min(nfit, nR), nj, J)
    c, err = fit(lam_, mu_, yv, dv, th); cu = cu_s; z = lambda k: torch.zeros(k, dtype=S.DT, device=DEV); a0, b0 = min(nfit, nL), min(nfit, nR)
    c = torch.cat([c[:a0], z(nL - a0), c[a0:a0 + b0], z(nR - b0), c[a0 + b0:], z(2 * NB + NFAR)])
else:
    c, err = fit(lam_, mu_, yv, dv, th)
print(f"  fit of the start ({'INIT' if init else 'family peak %.1f' % x['peak']}) on |d| > {DEX}: max err {err:.1e}", flush=True)
p = torch.cat([torch.tensor([lam_, mu_], dtype=S.DT, device=DEV), c])
r = C.residual(cu, p, G); print(f"  start residual sup {r[:G.y.numel()].abs().max().item():.2e} cont {r[-2].item():.1e} origin {r[-1].item():.1e}", flush=True)

LAMFIX = os.environ.get("LAMFIX")
if LAMFIX:       # lam held at a scan value: unknowns mu, c only (removes the strongly nonlinear lam direction)
    p[0] = float(LAMFIX); idx = torch.arange(1, p.numel(), device=DEV); fr = (1,); print(f"lam FIXED at {p[0].item():.10f}")
else:
    idx = torch.arange(p.numel(), device=DEV); fr = (0, 1)
p, hist = S.tsvd_gn(lambda q: C.res_jac(cu, q, G, free=fr), lambda q: C.residual(cu, q, G), p, idx, max_iter=int(os.environ.get("ITERS", 30)),
                    rc=float(os.environ.get("RC", 1e-12)), rc0=1e-6, tol=1e-14, verbose=True, floor=1e-9)
r = C.residual(cu, p, G); sup = r[:G.y.numel()].abs().max().item()
G2 = C.CuspGrid(cu, p[0].item(), R_src=RS * 1.37, m=MM + 1, h0=H0 * 0.8, off=0.37); r2 = C.residual(cu, p, G2); off = r2[:G2.y.numel()].abs().max().item()
lam, mu = p[0].item(), p[1].item(); u0 = 1 + lam - lam / mu; a = cu.standard(p)
# diagnostics at the cusp
dd = torch.tensor([0.0, -1e-10, 1e-10, -1e-6, 1e-6, -1e-3, 1e-3], dtype=S.DT, device=DEV); yy = yst + dd
U, V, Th = C.UV_at(cu, p, G, yy, dd); U2, V2, _ = C.UV_at(cu, p, G2, yy, dd)
Om = cu.fields(lam, mu, yy[1:], dd[1:])[1] @ p[2:]; kap = lam * abs(Th[0].item())
amps = (-Om * dd[1:].abs().sqrt()).tolist()
print(f"RESULT {tag}: lam {lam:.12f} mu {mu:.12f} U'(0) {u0:.12f} y*_std {a*yst:.10f} (scale {a:.10f})  sup {sup:.2e} OFF {off:.2e} cont {r[-2].item():.1e} orig {r[-1].item():.1e}  t={time.time()-t0:.0f}s")
print(f"  V(y*) {V[0].item():+.2e} (grid2 {V2[0].item():+.2e});  V(y*-+1e-10) {V[1].item():.3e} {V[2].item():.3e};  2 sqrt(kappa/2) 1e-5 = {2*math.sqrt(kap/2)*1e-5:.3e}")
print(f"  Theta* {Th[0].item():.10f} kappa {kap:.10f} sqrt(kappa/2) {math.sqrt(kap/2):.8f}; A(d) = -Omega sqrt|d| at d = -+1e-10, -+1e-6, -+1e-3: " + " ".join(f"{v:.8f}" for v in amps))
far1 = (G.d.abs() > 1e-3); far2 = (G2.d.abs() > 1e-3)
sup_far, off_far = r[:G.y.numel()][far1].abs().max().item(), r2[:G2.y.numel()][far2].abs().max().item()
print(f"  rows with |d| > 1e-3 only: sup {sup_far:.2e} OFF {off_far:.2e}")
Vd = torch.tensor([-1e-12, 1e-12, -1e-8, 1e-8], dtype=S.DT, device=DEV); Vv = C.UV_at(cu, p, G, yst + Vd, Vd)[1]
print("  V(d) / (2 sqrt(kappa/2) sqrt|d|) - 1 at d = -+1e-12, -+1e-8: " + " ".join(f"{(v / (2 * math.sqrt(kap / 2) * math.sqrt(abs(dd_)))) - 1:+.2e}" for v, dd_ in zip(Vv.tolist(), Vd.tolist())))
ia = r[:G.y.numel()].abs().argmax().item(); ib = r2[:G2.y.numel()].abs().argmax().item()
print(f"  worst row: y {G.y[ia].item():.6g} d {G.d[ia].item():+.3e};  off-grid worst: y {G2.y[ib].item():.6g} d {G2.d[ib].item():+.3e}")
aL, aR = C.amp_rows(cu, lam); rho_c = ((aR @ p[2:]) / (aL @ p[2:])).item(); print(f"  A_R/A_L at y* (from the coefficients) {rho_c:.10f}")
cL, cR = p[2:2 + nL].abs(), p[2 + nL:2 + nL + nR].abs()
print("  |cL| by k/8: " + " ".join(f"{cL[(i*nL)//8:((i+1)*nL)//8].max().item():.0e}" for i in range(8)) + "   |cR|: " + " ".join(f"{cR[(i*nR)//8:((i+1)*nR)//8].max().item():.0e}" for i in range(8)))
if NB or NFAR:
    o0 = cu.n0 + 2; print(f"  cusp blocks |cbL| {p[o0:o0+NB].abs().tolist() if NB <= 4 else p[o0:o0+NB].abs().max().item()}  far block max {p[o0+2*NB:].abs().max().item() if NFAR else 0:.2e}")
print(f"  |cL| tail {cL[-5:].max().item():.1e} |cR| tail {cR[-5:].max().item():.1e}  cL at n/2 {cL[nL//2-2:nL//2+2].max().item():.1e} cR at n/2 {cR[nR//2-2:nR//2+2].max().item():.1e}")
torch.save(dict(p=p.cpu(), ystar=yst, nL=nL, nR=nR, njJ=(nj, J), nb=NB, nfar=NFAR, lam=lam, mu=mu, sup=sup, off=off, scale=a), f"runs/sing_end_{tag}.pt")
with open("runs/sing_end.jsonl", "a") as f:
    f.write(json.dumps(dict(case="sing-end-" + tag, lam=lam, mu=mu, up0=u0, y_star_std=a * yst, y_star_raw=yst, nL=nL, nR=nR, njJ=[nj, J], nb=NB, nfar=NFAR,
                            R_src=RS, m=MM, h0=H0, res_sup=sup, res_offgrid=off, res_l2=r[:G.y.numel()].norm().item(), sup_far=sup_far, off_far=off_far, lam_fixed=LAMFIX, V_star=V[0].item(), lam_start=lam_, Theta_star=Th[0].item(), amps=amps, iters=len(hist), rho=rho_c, rho_imposed=C.RHO)) + "\n")
