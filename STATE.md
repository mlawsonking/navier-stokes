# STATE — read this first (builder / GPU session)

Last updated 2026-11-14 (evening). Branch `gpu-solver`. This file is the entry point for any future session:
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
| Non-smooth family mu(lam), a = -1 | mu = 2 at lam_1; min 1.9259 near 0.53; 2 at lam_2; max 2.01342 near 0.4594; 2.01103 at 0.45593 and falling | on = off about 1e-9 from lam = 0.515 down | NEW (same caveat); exponent law derived here, unchecked |
| 3rd unstable CCF profile lam_3 | NOT FOUND YET | family still at mu > 2 at lam = 0.4559 | 2511.22819 expects it below 0.455 |

Not established, do not quote: where the unstable a-branch ends. Hypothesis (derived 2026-11-14, UNTESTED):
at a = 0 the profile equation is the imaginary part of G + (1+lam) z G' - G^2/2 = 0 for G = U' - i Omega
(analytic in the upper half plane), so (1+lam) z G' = G(G-2)/2 and

    G = 2/(1 + c e^(-i pi alpha/2) z^alpha),   alpha = 1/(1+lam),  c > 0.

For 1 < alpha < 2 (lam in (-1/2, 0)) this has no pole in the closed upper half plane, has the right parity,
decays like y^(-alpha), and is NOT smooth at the origin (Omega ~ |y|^alpha). The smooth branch may converge
onto one member as a -> 0-: on the branch the peak of |Omega| grows without bound (58 at a = -0.11, 150 at
a = -0.08), the far-field amplitude q0 -> 0, and the peak moves toward the origin (normalisation
Omega'(0) = -0.1, so in peak units the origin slope -> 0).

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
Family below lam_2, looking for lam_3 (GPU, 5-7 min per step at n0 = 1024):

    SRC=runs/sing_track2_v6.pt .venv/Scripts/python -u runs/sing_track2.py 0.43 0.0005 v7 1024 8 4 0.0125

Watch `mu`: a smooth profile is where mu returns to 2. Then pin it (mu = 2 fixed, lam free) and refit with
the smooth basis only:

    .venv/Scripts/python -u runs/sing_smooth.py TRACK.pt LAM_NEAR N0 0.0125 runs/ccf_u3_a.pt
    .venv/Scripts/python -u runs/sing_to_smooth.py runs/ccf_u3_a.pt N0 0.0125 runs/ccf_u3_smooth.pt

Both of those still build "box" grids: switch them to `S.peak_grid` first (the box grid fails for peaks
this sharp).

First unstable branch in a (CPU, seconds per step):

    THREADS=6 EPSF=1.0 ACCEPT=3e-6 .venv/Scripts/python -u runs/gclm_log_branch.py runs/gclm_log_branch_w8.pt 0.3 0.002 w9 640

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
