# STATE — read this first (builder / GPU session)

Last updated 2026-11-16 (no jobs running). Branch `gpu-solver`. This file is the entry point for any future session:
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
| 2nd unstable CCF profile | lam_2 = 0.4713242278 (10 digits) | 4 discretisations within 6e-12; on = off 1e-9; smooth-only refit | agrees with 2511.22819 (0.47132422, read directly) and with the digits 2509.14185 marks as significant (0.47132; see 2a). Second solver `gclm_log` agrees to 5e-8, which is its own rounding floor here |
| Exact a = 1/2 solution | lam = -2/3 to 1e-15 | hand-derived closed form; in tests and `runs/gclm_log_check.py` | check of the general-a solvers |
| Stable branch lam_0(a) | about 10 digits, -1 <= a <= -0.3 | consistent with Lushnikov-Silantyev-Siegel Table 1 (3-4 digits) | old solver, not re-checked with on = off |
| 1st unstable branch lam_1(a) | -1.36 <= a <= -0.08; table in HANDOFF and `runs/gclm_u1_branch.jsonl` | on = off; 12 digits for a <= -0.5, degrading to about 6 near a = -0.08 | NEW as far as the literature agents found (not verified by me) |
| ... crosses lam_1 = 0 | at a = -0.41615 (interpolated) | representation regular through lam = 0 (`gclm_log.py`) | NEW (same caveat) |
| ... lam_1 + a nearly constant | -0.416 to -0.423 for -0.41 <= a <= -0.11 | observation | unexplained |
| Non-smooth family mu(lam), a = -1 | mu = 2 at lam_1; min 1.9259 near 0.53; 2 at lam_2; max 2.01342 near 0.4594; 2.01103 at 0.45593; 2.00689 at 0.454589 and falling | on = off about 1e-9 from lam = 0.515 down | NEW (same caveat); exponent law derived here, unchecked |
| 3rd unstable CCF profile lam_3 | NOT FOUND; may not lie on this family | family tracked to lam = 0.454544 (mu = 2.00662, peak 58.5, on = off 5e-8), where the tracker stalled (steps below 1e-5). On the last 9 points mu - 2 is linear in 1/peak with intercept +0.0022 at infinite peak, and lam -> 0.4535..0.4545 there: the data suggest the family ends in a singular profile BEFORE mu returns to 2. Suggestive only (extrapolation over a factor of 1.7 in peak) | 2511.22819 reports peak about 40 at 0.455 (mine 40.9 at 0.45495) and expects a third profile below 0.455 |

### 2a. Confirmation work done 2026-11-14 (night)
- lam_2 with the second solver (`runs/lam2_crosscheck.py`, ledger case `lam2-crosscheck-gclm_log`): the
  ccf_sing solution converted to the `gclm_log` representation (Omega agrees to 6.5e-12) and re-solved in
  vorticity form gives lam = 0.4713241768 (n = 640) and 0.4713241834 (n = 800, h = 0.01), i.e. 5e-8 and
  4e-8 below the ccf_sing value, with residual 1.3e-7 / 2.1e-7 (on = off) that Gauss-Newton cannot reduce.
  Reading: the vorticity form is one derivative higher than the Burgers form, so its rounding floor at this
  peak is about 1e-7 (the SAME function has Burgers residual 3.6e-10 and vorticity residual 4.6e-7). So this
  is a 7-digit confirmation only (0.4713242), not an independent 10-digit one. The 10 digits still rest on
  four ccf_sing discretisations, which share one equation form. Honest status: 7 digits confirmed two ways.
- Papers read directly by me (arXiv HTML, text search plus the results figure), replacing the agent reports:
  - 2511.22819: "inferred lam_s = 0.47132422"; no variation of their residual for lam in
    [0.471324222, 0.471324226]; peak "nearly 40 for lam about 0.455"; a non-smooth signal at the origin
    "remains for all tested lam in [0.455, 0.4713]"; a third profile, "if it exists", is at a sharper peak.
    All consistent with my family data (non-smooth throughout, peak 40.9 at 0.45495).
  - 2509.14185: body text says "lam_2 = 0.4703 with residuals of order 1e-7" (0.4703 is presumably a typo).
    The table in Fig. 2f prints 0.4713248638620 with only "0.47132" in bold, and the caption says the
    significant digits are the bold ones. So there is NO disagreement with that paper: the digits that
    differ from mine are ones the authors do not claim. The earlier "disagrees in the 7th digit" is withdrawn.
    For lam_1 they bold 0.60573370120, all of which I reproduce.
  - 2603.25104 (Huang, Tong, Wang): Thm 2.4 as reported (a = 0, odd degenerate data, c_l = 1/2, c_omega = -1,
    limit -pi(delta(X-1) - delta(X+1)), H = 2/(1-X^2)). Thm 2.6: for a < 0, HALF-LINE case, exact profile
    -sin(pi mu) 1{X>1} |1-X|^(-mu) ... with mu = 1/(1-a), c_l = 1-a (my lam = -a). For the ODD case they have
    numerics only (Table 5.1/6.1, dynamic rescaling, about 3 digits), gamma = 1 + lam:
        a      -0.1    -0.2    -0.3    -0.4    -0.5    -0.6    -0.7    -0.8    -0.9    -1.0
        gamma  0.7827  0.9114  1.0124  1.0986  1.1821  1.2782  1.3888  1.4858  1.5866  1.6841
        mine   0.6785  0.7759  0.8783  0.9843  1.0868  1.1904  1.2941  1.3979  1.5018  1.6057   (1 + lam_1)
    and gamma = 1/2 at a = 0 (theorem). Their singular odd family lies 0.08 to 0.14 ABOVE my smooth branch
    for a <= -0.1, but drops to 0.5 at a = 0 while my branch is at 0.652 at a = -0.0712, so the two curves
    must cross in (-0.1, 0).
- HYPOTHESIS (CONTRADICTED the next day, see 2b): the first unstable smooth branch ends where it meets the Huang-Tong-Wang
  singular odd family, i.e. at the a_c where gamma_HTW(a_c) = 1 + lam_1(a_c). My fit (a_c = -0.037, lam_1
  about -0.375, gamma about 0.625) is in the range where that crossing has to be, and the shape of the end
  (peak mass / y* tending to a constant, profile vanishing inside the peak relative to it, Omega'(0) -> 0 in
  the peak's own units) is what merging into a delta-at-X=1 profile would look like. Test: get gamma_HTW for
  -0.1 < a < 0 (dynamic rescaling of odd degenerate data; they tabulate nothing there) and see whether it
  passes through (-0.037, 0.625).

### 2b. Branch end revisited 2026-11-15: the 2a hypothesis is CONTRADICTED; "ends strictly before a = 0" is WITHDRAWN
- Fingerprint test on the last solution, a = -0.0712 (`runs/branch_end_fingerprint.py`). The HTW a < 0 singular
  profile would need a one-sided power law (X-1)^(-mu) and V = (1+lam) y + a U = 0 at the front. Found instead:
  the peak is a SYMMETRIC Lorentzian (half-widths left/right 2.89e-3 / 2.94e-3 of y*), U' is antisymmetric
  about it (+111 just inside, -109 just outside), and V at the peak is 0.49 (1+lam) y*, not zero. Away from the
  peak U'(y)/U'(0) follows 1/(1 - X^2), X = y/y*: 1.92 vs 1.96 at X = 0.7, 5.0 vs 5.26 at 0.9, -4.48 vs -4.76
  at 1.1, -1.40 vs -1.45 at 1.3. So the profile is a pair of simple poles close to the real axis at X = +-1,
  i.e. a smoothed version of the HTW a = 0 delta-pair profile (their Thm 2.4), NOT their a < 0 family.
- Fits redone (`runs/branch_end_fit2.py`): a finite end point q ~ (a_c - a)^e against an essential singularity
  at a = 0, q ~ |a|^p exp(-C/|a|) (same number of parameters). The exponential form fits as well or better in
  every window (FWHM/y*: rms 1.8e-4..4.4e-4 vs 3.2e-4..1.7e-3; 1/peak: 3e-5..3e-4 vs 4e-5..1.6e-3), its
  constant is stable (C = 0.193..0.207 for the width) while the power-law a_c drifts toward 0 as the window
  moves to the end (-0.043, -0.039, -0.0365, -0.035), and fitted on -0.13 < a < -0.085 it predicts the last
  point better (log FWHM/y*: actual -5.141, exponential -5.137, power law -5.172). A drifting a_c is what an
  essential singularity looks like through a power-law fit.
- Mechanism (heuristic, mine, unchecked): near a pole of residue r the a = 0 identity gives r = -2 V*, V* the
  transport speed at the pole. For a < 0, V* = (1+lam) + a U(1) in X units and U(1) = (U'(0)/2) log(2/eps)
  for a pole pair at distance eps from the axis, so log(2/eps) = 2 (gamma - U'(0)/4) / (|a| U'(0)): the pole
  distance is exponentially small in 1/|a|. Check at a = -0.0712: measured r = -0.70..-0.77 against
  -U'(0)/2 = -0.77 and -2 V* = -0.64; predicted C = 0.29 against fitted 0.20..0.26. Right size, not exact.
- Current reading: the branch most likely continues to a -> 0- with an exponentially thin spike and tends to
  the delta-pair profile there; a finite end point near -0.037 is NOT excluded by the data but is no longer
  the preferred description. Neither is established. What would settle it: follow the branch further (each
  step now costs a factor in resolution), or an asymptotic analysis of the pole for small |a| by the math session.
- `runs/htw_rescale.py` (dynamic rescaling from odd degenerate data): reproduces the exact a = 0 scaling
  factors (c_l = 0.500001, c_w = -1.000000). At a = -0.1 the spike sharpens faster than the fixed grid can
  follow (gamma 0.628 -> 0.697 by tau = 5.5, then under-resolved): no usable value. Not pursued.

### 2c. Small-|a| asymptotics of the first unstable branch (2026-11-16): heuristic derivation + numerical support
Derivation (mine, formal, NOT checked by the math session). X = y/y*, gamma = 1 + lam, G = U' - i Omega.
Away from the spikes Omega is negligible, so G is a pole pair, G = u0/(1 - z^2), u0 = U'(0), poles a distance eps
below X = +-1. Expand F = G + V G' - G^2/2 (V = gamma X + a U, U = (u0/2) log|(1+X)/(1-X)|) at the pole z = 1:
  1. origin (exact, smooth profile with Omega'(0) != 0):   u0 = (1 + gamma)/(1 - a)
  2. double-pole terms:   V* = u0/4,  V* = gamma + a (u0/2) log(2/eps)   (transport speed at the pole)
  3. simple-pole terms:   -u0/2 + (gamma + a u0/4) u0/2 + u0^2/8 = 0,  i.e.  gamma + a u0/4 = 1 - u0/4
(at a = 0 with u0 free these give u0 = 2, gamma = 1/2: the Huang-Tong-Wang delta-pair profile; relation 1 is
what the smooth branch has instead of u0 = 2.) Relations 1 and 3 give a closed form with no fitted constant:
      lam_1(a) ~ -2 (1 + a)/(5 - 3 a),      so  lam_1 -> -2/5,  U'(0) -> 8/5  as a -> 0-,
and relation 2 gives the spike half-width  eps ~ const * exp(-kappa/|a|),  kappa = 2(gamma - u0/4)/u0 -> 1/4.
Numerical test (`runs/branch_end_asymptotics.py`, all 115 solutions of `gclm_log_branch_w10.pt`):
- gamma - gamma_pred = +0.185 at a = -0.413, +0.0645 at -0.202, +0.0181 at -0.100, +0.0082 at -0.0712: it
  vanishes like 1.6..1.8 a^2 (one-parameter fit 1.78 a^2, rms 4.9e-4 on a > -0.12).
- Cubic extrapolation of the computed gamma to a = 0 gives 0.599 (prediction 0.600). Allowing a free constant in
  the difference gives -0.007, so the data pin the limit to about 0.60 +- 0.01.
- |a| log(2/eps) measured exceeds 2(gamma - u0/4)/u0 by 1.3..1.7 times |a| (0.465 vs 0.346 at -0.0712), i.e.
  eps is about 0.2 times exp(-kappa/|a|) x 2: an O(1) prefactor, as a leading-log relation allows. The fitted
  C = 0.20..0.27 of section 2b brackets kappa = 1/4.
- peak x eps -> u0/2 is approached slowly from below (0.51 at -0.41, 0.66 at -0.0712, target 0.77).
Status: a conjecture with a formal argument and a parameter-free match (three digits in the limit, error
shrinking like a^2). The weak point of the argument is relation 3: U has a logarithm at the pole and
|a| log(1/eps) = O(1), so the claim that the log terms do not feed into the simple-pole balance at leading order
needs a proper matched expansion (inner region zeta = (X-1)/eps, intermediate region where
Omega ~ V(X)^(-1/|a|)). That is the question for the math session.
This also explains the old observation "lam_1 + a nearly constant": -2(1+a)/(5-3a) + a + 1.8 a^2 is flat near
-0.42 over -0.4 < a < -0.1.

Not established, do not quote: where the unstable a-branch ends. What is known (2026-11-14, late):
- Data: the branch is followed to a = -0.0712 (lam_1 = -0.34807). The peak of |Omega| is a narrowing
  Lorentzian: height 42 -> 225 for a = -0.126 -> -0.071, position y* 1.28 -> 0.47, height x FWHM / y*
  nearly constant (1.20 -> 1.32), far-field amplitude q0 -> 0 (-0.247 -> -0.018), lam_1 + a = -0.42,
  U'(0) = 1.51 -> 1.54.
- Fit of the end point (`runs/branch_end_fit.py`, tail a > -0.10, power laws in a_c - a): FWHM/y* gives
  a_c = -0.0365 (exponent 1.71), 1/peak gives -0.0350 (1.85), q0 gives -0.0380 (2.77), y* gives -0.0345
  (1.13). Forcing a_c = 0 is 10 to 30 times worse in rms for every quantity. So the branch appears to end at
  a_c = -0.037 +- 0.005, STRICTLY before a = 0 [WITHDRAWN as a conclusion, see 2b], with the peak collapsing onto the origin (y* -> 0 about
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
- Literature (arXiv:2603.25104, now read directly, see 2a): their Thm 2.4 says that at a = 0,
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
   Read directly it turned out the "disagreeing" digits were ones that paper marks as not significant.
12. The vorticity-form solvers (`ccf_fast`, `gclm_log`) have a rounding floor about 1e3 times higher than the
    Burgers form (`ccf_sing`) at a sharp peak (1e-7 at lam_2): they cannot confirm more than 7 digits there.
9. The absolute residual floor grows like peak^2/width (rounding), so a fixed absolute threshold stalls a
   branch whose peak is diverging. Use a relative threshold together with on = off.
10. A lower bound on the map width (`EPSMIN`, 0.004 then 0.002 in old scripts) silently under-resolves a
    sharpening peak. Check the printed `eps` is not sitting on its floor.
11. With a very sharp peak the truncated-SVD step is only reliable from an excellent predictor; a looser
    cutoff (1e-6) is more robust but stalls at 1e-6. `tsvd_gn` starts at `rc0` and tightens to `rc`.

## 6. Open items, in order (superseded where section 7 lists a TODO)
1. lam_3: continue the family; pin; verify at two or more resolutions with on = off; smooth-only refit.
2. Branch end: test the a = 0 hypothesis (compare the profile near a = -0.05 with the explicit family; fit alpha).
3. DONE (7 digits only, see 2a). A genuinely independent 10-digit check of lam_2 would need a second
   Burgers-form discretisation (e.g. different basis or a finite-difference/Newton solve), not `gclm_log`.
4. DONE (see 2a). New item from it: test the branch-end hypothesis by computing gamma_HTW(a) for
   -0.1 < a < 0 (dynamic rescaling of the gCLM equation from odd data with Omega'(0) = 0).
5. Questions for the math session: end of the fourth update in HANDOFF.md, plus the a = 0 derivation above.
6. Unexplained: residual floor about 6e-10 at lam_2 with coefficients at 1e-14; the 1e-6 mid-spectrum
   plateau on the a-branch for a > -0.6 (possibly an inner layer at the origin as the branch nears its end).
7. Later: rung 2b (2D Boussinesq) in a separate Opus session.

## 7. Where this stands as a contribution (updated 2026-11-16)
Stages toward a short numerical note: (1) solver, (2) reproduce known values, (3) new numerical facts and
(4) confirmation pass are done as far as this session can take them; (5) write-up has a first draft, `NOTE.md`
(repo root; tables from `runs/make_note_tables.py`); (6) posting not started and needs Michael.
The most interesting content is now section 5 of the note: the small-|a| law lam_1 ~ -2(1+a)/(5-3a).
Open before the note could be shown to anyone (all marked [TODO] in NOTE.md):
- math-session check of the pole argument (HANDOFF fifth update) and of the exponent law (4);
- DONE 2026-11-16: family between lam_1 and lam_2 recomputed with the current tracker (`sing-track2-x1`,
  `runs/sing_track2_x1.pt`; on = off <= 4e-9; mu min 1.92587 at 0.53). The OLD files `runs/sing_track_t*.pt` cannot be
  evaluated with the current `ccf_sing` (residual O(1): the ansatz changed since) and `t2` was wrong at lam = 0.50, 0.49
  (1.94353, 1.95471 instead of 1.94843, 1.96470): do not use them. Branch rows a <= -1 checked off-grid
  (`runs/branch_offgrid_check.py`): 6e-9 .. 1e-6, so 7-8 digits there, not 12. Still unchecked: a = -0.9 only;
- a human literature check of novelty; number of unstable directions along the a-branch;
- optional: independent Burgers-form check of lam_2 beyond 7 digits; branch closer to a = 0.
This is not a result about Navier-Stokes. No third profile was found.
