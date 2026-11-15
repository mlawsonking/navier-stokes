# STATE — read this first (builder / GPU session)

Last updated 2026-11-14 (late evening). Branch `gpu-solver`. This file is the entry point for any future session:
what is established, what is only provisional, what is running, how to resume, and what burned us.
`HANDOFF.md` is the chronological log with the full tables and every withdrawn claim; `runs/README.md`
lists the scripts and data files.

## 1. What the project is doing right now
Rung 2a of the ladder: self-similar blow-up profiles of the 1D generalised Constantin-Lax-Majda family

    omega_t + a u omega_x = u_x omega,  u_x = H omega      (a = -1: Cordoba-Cordoba-Fontelos, CCF)
    omega = (T-t)^-1 Omega(x/(T-t)^(1+lam))
    Omega + ((1+lam) y + a U) Omega' - Omega U' = 0,  U' = H Omega

lam is solved as an unknown. Rung 2b (2D Boussinesq) has not been started.

## 2. Results and how far each can be trusted
"on = off" means the residual on a second, denser, differently placed grid equals the fit residual
(the only acceptance test now in use, see section 5).

| Result | Value | Evidence | Status |
|---|---|---|---|
| Stable CCF profile | lam_0 = 1.1807776628999 | published 1.1807776628998 | reproduces literature |
| 1st unstable CCF profile | lam_1 = 0.605733701203 (+-2e-12) | published 0.6057337012032; two solvers, two equation forms | reproduces literature |
| 2nd unstable CCF profile | lam_2 = 0.4713242278 (10 digits) | 4 discretisations within 6e-12; on = off 1e-9; smooth-only refit | agrees with 2511.22819 (0.47132422); disagrees with 0.4713248638620 in the 7th digit |
| Exact a = 1/2 solution | lam = -2/3 to 1e-15 | hand-derived closed form; in tests and `runs/gclm_log_check.py` | check of the general-a solvers |
| Stable branch lam_0(a) | about 10 digits, -1 <= a <= -0.3 | consistent with Lushnikov-Silantyev-Siegel Table 1 (3-4 digits) | old solver, not re-checked with on = off |
| 1st unstable branch lam_1(a) | -1.36 <= a <= -0.08; table in HANDOFF and `runs/gclm_u1_branch.jsonl` | on = off; 12 digits for a <= -0.5, degrading to about 6 near a = -0.08 | NEW as far as the literature agents found (not verified by me) |
| ... crosses lam_1 = 0 | at a = -0.41615 (interpolated) | representation regular through lam = 0 (`gclm_log.py`) | NEW (same caveat) |
| ... lam_1 + a nearly constant | -0.416 to -0.423 for -0.41 <= a <= -0.11 | observation | unexplained |
| Non-smooth family mu(lam), a = -1 | mu = 2 at lam_1; min 1.9259 near 0.53; 2 at lam_2; max 2.01342 near 0.4594; 2.01103 at 0.45593; 2.00689 at 0.454589 and falling | on = off about 1e-9 from lam = 0.515 down | NEW (same caveat); exponent law derived here, unchecked |
| 3rd unstable CCF profile lam_3 | NOT FOUND; may not lie on this family | family tracked to lam = 0.454544 (mu = 2.00662, peak 58.5, on = off 5e-8), where the tracker stalled (steps below 1e-5). On the last 9 points mu - 2 is linear in 1/peak with intercept +0.0022 at infinite peak, and lam -> 0.4535..0.4545 there: the data suggest the family ends in a singular profile BEFORE mu returns to 2. Suggestive only (extrapolation over a factor of 1.7 in peak) | 2511.22819 reports peak about 40 at 0.455 (mine 40.9 at 0.45495) and expects a third profile below 0.455 |

Not established, do not quote: where the unstable a-branch ends. What is known (2026-11-14, late):
- Data: the branch is followed to a = -0.0712 (lam_1 = -0.34807). The peak of |Omega| is a narrowing
  Lorentzian: height 42 -> 225 for a = -0.126 -> -0.071, position y* 1.28 -> 0.47, height x FWHM / y*
  nearly constant (1.20 -> 1.32), far-field amplitude q0 -> 0 (-0.247 -> -0.018), lam_1 + a = -0.42,
  U'(0) = 1.51 -> 1.54.
- Fit of the end point (`runs/branch_end_fit.py`, tail a > -0.10, power laws in a_c - a): FWHM/y* gives
  a_c = -0.0365 (exponent 1.71), 1/peak gives -0.0350 (1.85), q0 gives -0.0380 (2.77), y* gives -0.0345
  (1.13). Forcing a_c = 0 is 10 to 30 times worse in rms for every quantity. So the branch appears to end at
  a_c = -0.037 +- 0.005, STRICTLY before a = 0, with the peak collapsing onto the origin (y* -> 0 about
  linearly) while its relative width -> 0 and its height diverges; lam_1 extrapolates to about -0.375 there.
  Exponents are not clean (they drift with the fitting window): provisional.
- Derived here (for the math session to check): at a = 0 the profile equation is the imaginary part of
  G + (1+lam) z G' - G^2/2 = 0, G = U' - i Omega analytic in the upper half plane, so
  (1+lam) z G' = G(G-2)/2 and G = 2/(1 - C z^alpha), alpha = 1/(1+lam). A smooth odd profile needs alpha = 1,
  i.e. lam = 0: the classical CLM profile is the ONLY smooth self-similar profile at a = 0. So the unstable
  branch (lam_1 near -0.4 there) cannot reach a = 0 as a smooth profile: it must end, or become singular,
  at some a_c <= 0. For 1 < alpha < 2, G = 2/(1 + c e^(-i pi alpha/2) z^alpha) is an explicit profile that is
  not smooth at the origin, with peak |Omega| = tan(pi alpha/4), which diverges as alpha -> 2 (lam -> -1/2).
- My first guess (branch converges onto a member of that family with alpha near 1.7) is REFUTED by the
  amplitude: that member has peak 4.5, the branch has 225 and growing.
- Literature lead (agent report on arXiv:2603.25104, NOT verified by me): their Thm 2.4 says that at a = 0,
  odd degenerate data give the limit Omega = -pi(delta(X-1) - delta(X+1)), H Omega = 2/(1 - X^2), with their
  gamma = 1/2 (my lam = -1/2) — the alpha -> 2 end of the family above; and for a < 0 their numerics show
  odd profiles singular at X = +-1 (my lam about -0.22 at a = -0.1, a different object from my branch,
  which has lam_1 = -0.32 there). They state no classification that would exclude my branch. A delta at X = 1
  of mass pi, smoothed, has height x FWHM = 2 in units with U'(0) = 2; my branch has 1.29 x (2/1.54) = 1.68
  and rising. Consistent with a delta-type limit, not established.

## 3. Code map (`gpu_solver/`)
- `ccf.py` — `Grid` (sinh grid, optional Lorentzian cluster), old autodiff residual.
- `ccf_fast.py` — Omega-form solver for general a: Theta = r^beta M(s), sinh-mapped Chebyshev basis (`MAP`),
  analytic Jacobian, `lm` (damped) and `gn` (truncated-SVD Gauss-Newton). Fails as lam -> 0.
- `gclm_log.py` — same equation, Theta = r^beta s N(s) + q0 (r^beta - 1)/(2 beta), regular through lam = 0.
  Use this for the a-branch.
- `ccf_sing.py` — a = -1 only, Burgers form, origin ansatz with unknown exponent mu; `EGrid` (exponential
  grid; clusters "box" and "sinh"); `peak_grid` (rows follow the basis density, at least R rows per basis
  function); `tsvd_gn`; `lm` (kept; do not use near mu = 2).
- `burgers.py`, `clm.py`, `hilbert.py`, `lm.py`, `batch_lm.py` — rung 1.

## 4. How to resume (Git Bash, repo root, `export PYTHONPATH=.`)
Family below lam_2, looking for lam_3 (GPU; 2-9 min per step at n0 = 1024, steps of 3e-5 to 5e-5 in lam):

    SRC=runs/sing_track2_v10.pt EPSMIN=0.00004 ACCEPT=3e-7 .venv/Scripts/python -u runs/sing_track2.py 0.44 0.00005 v11 1024 8 4 0.0125

(use the newest `runs/sing_track2_v*.pt` that exists; v8 is the last one committed). Watch `mu`: a smooth
profile is where mu returns to 2. Then pin with the smooth basis only, lam unknown:

    .venv/Scripts/python -u runs/sing_pin2.py TRACK.pt runs/ccf_u3.pt 1024 0.0125

Tried and failed at mu = 2.007 (do not repeat without a new idea): `runs/sing_mu_march.py` (march in mu with
lam unknown: Newton diverges even for d mu = 1e-4) and `sing_pin2.py` straight from mu = 2.007 (start
residual 0.2). At this peak sharpness Gauss-Newton only converges from the 3-point predictor of the lam-march.
`sing_smooth.py` and `sing_to_smooth.py` are the older "box"-grid versions (fine for lam_2, not for lam_3).

First unstable branch in a (CPU, seconds per step):

    THREADS=6 EPSF=1.0 ACCEPT=1e-8 .venv/Scripts/python -u runs/gclm_log_branch.py runs/gclm_log_branch_w10.pt 0.3 0.001 w11 640

(ACCEPT is now RELATIVE to peak^2/width; the run stopped at a = -0.0712 on that threshold, about 100 s per step.)

Checks:

    .venv/Scripts/python runs/gclm_log_check.py         exact a = 1/2 and conversion
    .venv/Scripts/python runs/quad_sinh_grid_check.py   quadrature on exact sharp Hilbert pairs
    .venv/Scripts/python runs/sing_offgrid_check.py TRACK.pt [modulus]
    tests/test_gpu_solver.py: import it and call the test_ functions (pytest is not installed)

Process hygiene: never kill by image name (other sessions run Python); find the PID by command line.
Do not run two GPU jobs at once (one CUDA illegal-memory-access crash on 2026-11-14 with two running, and a
forgotten second tracker slowed everything for an hour).

## 5. Traps that produced wrong numbers here (each one cost a retraction)
1. The residual at the fit rows is NOT the residual. A peak-concentrated basis can be 1e-11 at its rows and
   0.2 between them. Accept only on = off.
2. Rows must outnumber basis functions LOCALLY (at least 6 per function), not just in total. Use `peak_grid`.
3. A cluster that is dense only in a narrow core (Lorentzian) or only inside a box leaves the peak's
   shoulders under-sampled; the off-grid residual then sits on the shoulder.
4. The sinh-grid Lorentzian cluster is centred by iteration (`tp`); the first version was centred off the peak.
5. Damped least squares (LM) crawls or stalls when the Jacobian is numerically rank-deficient (singular
   blocks near mu = 2; lam -> 0). Truncated-SVD Gauss-Newton converges quadratically. It must be a real SVD
   on the CPU: QR-Tikhonov with the same cutoff does not converge, and the CUDA SVD hung once.
6. Predictors must be built in coordinates where the solution moves slowly: coefficients relative to the
   peak-aligned map, log of the map width, and beta*c (not c) in the old representation as lam -> 0.
7. Too narrow a sinh map (eps = 0.33 FWHM) gives a coefficient plateau and 7th-digit errors; eps near the
   FWHM is safe.
8. Sub-agent literature reports are leads, not facts. Two of them disagree about what 2509.14185 says.
9. The absolute residual floor grows like peak^2/width (rounding), so a fixed absolute threshold stalls a
   branch whose peak is diverging. Use a relative threshold together with on = off.
10. A lower bound on the map width (`EPSMIN`, 0.004 then 0.002 in old scripts) silently under-resolves a
    sharpening peak. Check the printed `eps` is not sitting on its floor.
11. With a very sharp peak the truncated-SVD step is only reliable from an excellent predictor; a looser
    cutoff (1e-6) is more robust but stalls at 1e-6. `tsvd_gn` starts at `rc0` and tightens to `rc`.

## 6. Open items, in order
1. lam_3: continue the family; pin; verify at two or more resolutions with on = off; smooth-only refit.
2. Branch end: test the a = 0 hypothesis (compare the profile near a = -0.05 with the explicit family; fit alpha).
3. Independent confirmation of lam_2 (and lam_3 if found) with `gclm_log.py` at a = -1: different equation
   form and representation.
4. Read 2511.22819, 2509.14185, 2603.25104 directly (figures and tables) instead of through agents.
5. Questions for the math session: end of the fourth update in HANDOFF.md, plus the a = 0 derivation above.
6. Unexplained: residual floor about 6e-10 at lam_2 with coefficients at 1e-14; the 1e-6 mid-spectrum
   plateau on the a-branch for a > -0.6 (possibly an inner layer at the origin as the branch nears its end).
7. Later: rung 2b (2D Boussinesq) in a separate Opus session.

## 7. Where this stands as a contribution (assessment given to Michael on 2026-11-14)
Mid-stage 3 of 6 toward a short numerical note: (1) solver and (2) reproduce known values are done;
(3) new numerical facts is partly done; (4) independent confirmation, (5) write-up, (6) posting are not
started. This is not a result about Navier-Stokes. A third profile would make the note clearly worth posting.
