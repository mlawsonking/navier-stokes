# Unstable self-similar blow-up profiles of the generalised Constantin-Lax-Majda model: a branch in the advection parameter, a non-smooth family, and a small-|a| limit

DRAFT 1, 2026-11-16. Working note, not submitted anywhere. Every number below was computed in this repository
(branch `gpu-solver`); the run ledgers are `runs/*.jsonl`, and `STATE.md` records what is trusted and what was
withdrawn. Tables are produced by `runs/make_note_tables.py`. Items marked [TODO] are not done.

## Summary

For the generalised Constantin-Lax-Majda (gCLM, Okamoto-Sakajo-Wunsch) equation

    omega_t + a u omega_x = u_x omega,    u_x = H omega,

we compute exact self-similar blow-up profiles omega = (T-t)^-1 Omega(x/(T-t)^(1+lam)) with a spectral
collocation method in which lam is an unknown. At a = -1 this is the Cordoba-Cordoba-Fontelos (CCF) model.
We report:

1. The second unstable smooth CCF profile: lam_2 = 0.4713242 (seven digits confirmed by two solvers using
   different equation forms; 0.4713242278 from the more accurate one). Consistent with the published 0.47132
   (Wang et al. 2025, significant digits) and 0.47132422 (arXiv:2511.22819).
2. The continuation of the FIRST unstable profile in the advection parameter, lam_1(a), for
   -1.36 <= a <= -0.071. The exponent decreases almost linearly, passes through lam_1 = 0 at a = -0.41615,
   and is negative beyond. We have not found this branch in the literature. [TODO: literature check by a person]
3. A formal small-|a| description of that branch, with no fitted constant:
   lam_1(a) = -2(1+a)/(5-3a) + O(a^2), so lam_1 -> -2/5 as a -> 0-, while the profile tends to a pair of
   spikes (the delta-pair profile that Huang, Tong and Wang prove at a = 0) whose width is exponentially small
   in 1/|a|. The computed branch agrees: the difference is 1.8 a^2 and the extrapolated limit is -0.401.
   This is a conjecture supported by numerics, not a theorem.
4. At a = -1, the one-parameter family of NON-smooth profiles, Omega ~ |y|^(mu-1) sgn(y) at the origin, that
   connects the smooth ones: the exponent mu(lam) between lam_1 and lam_2 and below lam_2 down to
   lam = 0.45454, where the peak of the profile has grown to 58 and our solver stalls. No third smooth profile
   was found; on the computed range mu stays above 2 and the data suggest (weakly) that the family becomes
   singular before mu returns to 2.

Nothing here is a statement about the Navier-Stokes or Euler equations.

## 1. Equations and normalisation

Profile equation (y is the self-similar variable, U' = H Omega, U(0) = 0, Omega odd):

    Omega + ((1+lam) y + a U) Omega' - Omega U' = 0.                                  (1)

Evaluating (1) at the origin for a smooth profile with Omega'(0) != 0 gives the exact relation

    lam = (1 - a) U'(0) - 2.                                                           (2)

At a = -1, with Theta' = Omega, (1) integrates once to the Burgers-type form

    -lam Theta + ((1+lam) y - H Theta) Theta' = 0.                                     (3)

Normalisation: Omega'(0) = -0.1 for the a-branch and for lam_2 (the profile is otherwise defined up to
y -> c y). The far field is Omega ~ q0 y^(-1/(1+lam)).

Non-smooth profiles at a = -1. If Theta ~ A |y|^mu at the origin, the leading balance of (3) gives

    mu = lam / (1 + lam - U'(0)),                                                      (4)

and the local expansion contains the exponents mu + j(mu - 1) + 2k (j, k >= 0). mu = 2 is a smooth profile.
[Derived in this project; not independently checked. TODO]

## 2. Method

- Representation. With r = 1 + y^2, alpha = 1/(1+lam), beta = (1-alpha)/2 and the compact variable
  s = r^(-alpha/2) in (0, 1], Theta = r^beta M(s), M a Chebyshev series in a sinh-mapped variable that
  concentrates resolution at the peak of |Omega| (map width about the full width at half maximum).
  For the a-branch a variant that stays regular through lam = 0 is used:
  Theta = r^beta s N(s) + q0 (r^beta - 1)/(2 beta) (`gpu_solver/gclm_log.py`).
  For non-smooth profiles: Theta = r^beta (1-s)^(mu/2) [N_0 + sum_j (1-s)^(j(mu-1)/2) N_j]
  (`gpu_solver/ccf_sing.py`).
- Hilbert transform. Trapezoid rule on a mapped grid with collocation at the midpoints (spectrally accurate
  for the principal value), on an exponential grid y = e^xi whose density follows the basis near the peak
  (at least six rows per basis function), with analytic tail corrections.
- Solve. Collocation residual of (1) or (3) plus the normalisation and the origin relation; unknowns are lam
  (or mu) and the coefficients; Gauss-Newton with a truncated SVD and an explicit Jacobian.
- Continuation. Three-point extrapolation in the parameter of the coefficients, each in its own peak-aligned
  map, and of the map parameters.
- Acceptance test. A solution is accepted only if its residual on a second grid (denser, shifted, wider) equals
  the residual on the fitting grid. This test was introduced after five separate cases in which a small
  fitting residual was produced by aliasing; those results were withdrawn (HANDOFF.md).
- Checks against known values: stable CCF profile lam_0 = 1.1807776628999 (published 1.1807776628998);
  first unstable lam_1 = 0.605733701203 (published 0.6057337012032, two solvers and both equation forms);
  the exact solution at a = 1/2 (lam = -2/3) to 1e-15.

## 3. The second unstable CCF profile

| discretisation | lam_2 |
|---|---|
| ccf_sing (Burgers form), n = 448 | 0.47132422779 |
| n = 640 | 0.47132422777 |
| n = 640 (file ccf_u2_smooth640c) | 0.47132422776 |
| n = 640 (file ccf_u2_smooth640h) | 0.47132422777 |
| n = 800 | 0.47132422777 |
| gclm_log (vorticity form), n = 640 | 0.4713241768 |
| gclm_log, n = 800, h = 0.01 | 0.4713241834 |

Peak of |Omega| 8.2387 at y = 7.722; U'(0) = 1.2356621139. The Burgers-form residual is 4e-10 on and off the
fitting grid. The vorticity-form solver cannot go below a residual of 1e-7 at this peak (it differentiates once
more), so it confirms seven digits only. Digits eight to ten rest on one equation form. [TODO: an independent
Burgers-form discretisation if ten digits are to be claimed.]

## 4. The first unstable branch in a

| a | lam_1 | -2(1+a)/(5-3a) | peak of abs(Omega) | residual | run |
|---|---|---|---|---|---|
| -1.3600 | +0.980312923015 | | 1.124 | 1.4e-06 | branch-down192b |
| -1.2000 | +0.813748227860 | | 1.293 | 8.2e-08 | branch-down192b |
| -1.1000 | +0.709713573142 | | 1.426 | 2.7e-08 | branch-down192b |
| -1.0000 | +0.605733701203 | | 1.590 | 5.6e-09 | branch-down192b |
| -0.9000 | +0.501810727477 | | | 6.7e-13 (on-grid only) | branch-verify |
| -0.8000 | +0.397947442081 | | 2.064 | 2.0e-11 | branch-upe256 |
| -0.7000 | +0.294148397709 | | 2.426 | 2.4e-11 | branch-upe256 |
| -0.6000 | +0.190422570934 | | 2.940 | 7.0e-11 | branch-upe256 |
| -0.5000 | +0.086790210136 | | 3.726 | 8.8e-11 | branch-upg256 |
| -0.4350 | +0.019500815257 | | 4.502 | 7.4e-10 | branch-uph256 |
| -0.4010 | -0.015664052888 | -0.1931 | 5.047 | 2.2e-09 | log-w1 |
| -0.2982 | -0.121717484902 | -0.2381 | 7.865 | 8.2e-09 | log-w3 |
| -0.1982 | -0.224114370317 | -0.2866 | 15.909 | 4.8e-08 | log-w3 |
| -0.1502 | -0.272484878468 | -0.3118 | 27.924 | 7.4e-08 | log-w3 |
| -0.1001 | -0.321455100895 | -0.3395 | 76.145 | 1.6e-07 | log-w5 |
| -0.0901 | -0.330867472846 | -0.3453 | 103.465 | 1.6e-07 | log-w7 |
| -0.0801 | -0.340082707460 | -0.3511 | 150.012 | 8.8e-07 | log-w8 |
| -0.0712 | -0.348060264810 | -0.3563 | 225.248 | 3.8e-04 | log-w10 |

Not all printed digits are significant: about twelve for a <= -0.5, falling to about five at the last point,
where the residual is accepted relative to its natural scale peak^2/width (the absolute rounding floor grows
with the peak). For a <= -1 the solutions (192 basis functions, fitted on the older grid) were checked afterwards on two
other grids (`runs/branch_offgrid_check.py`): residual 6e-9 at a = -1 growing to 1e-6 at a = -1.36, against
1e-13 on the fitting grid, so those rows are good to about seven or eight digits, not twelve (the a = -1 value
nevertheless matches the published one to twelve). The a = -0.9 row predates the test. [TODO]

lam_1 = 0 at a = -0.41615 (interpolated). For a between -0.41615 and 0, lam_1 < 0: the blow-up still focuses, but its spatial scale
(T-t)^(1+lam) shrinks more slowly than T-t.

## 5. The limit a -> 0-

Observation (a = -0.0712, `runs/branch_end_fingerprint.py`). With X = y/y*, y* the peak position:
the peak is a symmetric Lorentzian (half-widths 2.89e-3 and 2.94e-3 on the two sides), U' is antisymmetric
about it, and away from it U'(X)/U'(0) = 1/(1 - X^2) within a few percent (1.92 against 1.96 at X = 0.7;
-1.40 against -1.45 at X = 1.3). So G = U' - i Omega is close to a pair of simple poles,
G = u0/(1 - z^2), displaced a distance eps below the real axis.

Formal argument. At a = 0, (1) is the imaginary part of F = G + (1+lam) z G' - G^2/2 = 0, G analytic in the
upper half plane; a pole pair then requires u0 = 2 and 1 + lam = 1/2, the delta-pair profile of Huang, Tong
and Wang (arXiv:2603.25104, Theorem 2.4). For small a < 0 keep the pole pair as the outer solution, write
gamma = 1 + lam, V = gamma X + a U with U = (u0/2) log|(1+X)/(1-X)|, and balance the terms of F at the pole:

    origin relation (2):      u0 = (1 + gamma)/(1 - a)
    double-pole terms:        gamma + a (u0/2) log(2/eps) = u0/4
    simple-pole terms:        gamma + a u0/4 = 1 - u0/4

The first and third give

    lam_1(a) = -2 (1 + a)/(5 - 3 a) + ...,      lam_1 -> -2/5,   U'(0) -> 8/5,                 (5)

and the second gives eps = const * exp(-kappa/|a|) with kappa -> 1/4. The smooth branch therefore does not
converge to the degenerate a = 0 profile's exponent (-1/2): relation (2), which holds because Omega'(0) != 0,
replaces u0 = 2.

Numerical support (`runs/branch_end_asymptotics.py`, 115 computed solutions).

| a | 1 + lam_1 computed | (3-5a)/(5-3a) | difference | difference / a^2 |
|---|---|---|---|---|
| -0.4130 | 0.99674 | 0.81183 | +0.18492 | 1.08 |
| -0.3022 | 0.88240 | 0.76374 | +0.11866 | 1.30 |
| -0.2022 | 0.77995 | 0.71542 | +0.06453 | 1.58 |
| -0.1422 | 0.71955 | 0.68387 | +0.03568 | 1.76 |
| -0.1001 | 0.67854 | 0.66045 | +0.01809 | 1.80 |
| -0.0801 | 0.65992 | 0.64893 | +0.01098 | 1.71 |
| -0.0712 | 0.65193 | 0.64372 | +0.00821 | 1.62 |

The difference vanishes like 1.8 a^2; a cubic extrapolation of the computed values to a = 0 gives 0.599.
The spike width is fitted at least as well by |a|^p exp(-C/|a|), C = 0.20 to 0.27, as by a power law ending at
a finite a_c, and the fitted a_c drifts toward 0 as the fitting window moves toward the end
(`runs/branch_end_fit2.py`). The measured |a| log(2/eps) exceeds the double-pole relation by about 1.6 |a|,
an O(1) prefactor in eps.

Limits of this section. The argument is formal: U has a logarithm at the pole and |a| log(1/eps) is O(1), so
the simple-pole balance needs a matched expansion to be justified. The branch has been computed only to
a = -0.071; a finite end point between -0.07 and 0 is not excluded by the data. [TODO: math check]

## 6. The non-smooth family at a = -1

| lam | mu | U'(0) | peak of abs(Omega) | off-grid residual | run |
|---|---|---|---|---|---|
| 0.605734 (lam_1) | 2 | 1.302867 | 1.590 | | smooth |
| 0.600000 | 1.99123 | 1.298678 | 1.639 | 1.3e-11 | track2-x1 |
| 0.580000 | 1.96266 | 1.284482 | 1.841 | 1.1e-10 | track2-x1 |
| 0.560000 | 1.93989 | 1.271324 | 2.113 | 3.5e-10 | track2-x1 |
| 0.540000 | 1.92709 | 1.259785 | 2.499 | 9.2e-10 | track2-x1 |
| 0.530000 | 1.92587 | 1.254800 | 2.760 | 1.4e-09 | track2-x1 |
| 0.520000 | 1.92882 | 1.250404 | 3.088 | 1.9e-09 | track2-x1 |
| 0.510000 | 1.93631 | 1.246612 | 3.516 | 2.5e-09 | track2-x1 |
| 0.500000 | 1.94843 | 1.243383 | 4.096 | 3.0e-09 | track2-x1 |
| 0.490000 | 1.96470 | 1.240598 | 4.929 | 3.4e-09 | track2-x1 |
| 0.478750 | 1.98612 | 1.237703 | 6.463 | 3.6e-09 | track2-x1 |
| 0.473750 | 1.99565 | 1.236359 | 7.549 | 3.6e-09 | track2-x1 |
| 0.471324 (lam_2) | 2 | 1.235662 | 8.239 | 4e-10 | smooth |
| 0.467500 | 2.00615 | 1.234467 | 9.679 | 1.6e-09 | track2-v2 |
| 0.465000 | 2.00948 | 1.233596 | 10.992 | 2.6e-09 | track2-v2 |
| 0.461680 | 2.01257 | 1.232282 | 13.599 | 3.0e-09 | track2-v2 |
| 0.459430 | 2.01342 | 1.231246 | 16.517 | 4.0e-09 | track2-v3 |
| 0.457630 | 2.01298 | 1.230290 | 20.471 | 3.4e-08 | track2-v4 |
| 0.455930 | 2.01103 | 1.229216 | 28.247 | 6.3e-09 | track2-v6 |
| 0.454995 | 2.00868 | 1.228480 | 39.811 | 1.0e-08 | track2-v7 |
| 0.454769 | 2.00779 | 1.228267 | 46.102 | 9.0e-08 | track2-v7 |
| 0.454544 | 2.00662 | 1.228022 | 58.460 | 4.1e-08 | track2-v10 |

All rows pass the acceptance test (residual on the second grid equal to the fitting residual). The stretch
between lam_1 and lam_2 was recomputed on 2026-11-16 (run track2-x1); the first tracker's values there agreed
to nine digits for lam >= 0.52 and were wrong in the third decimal at lam = 0.50 and 0.49. The smooth profiles are exactly the points mu = 2.

Below lam_2 the exponent rises to 2.01342 near lam = 0.4594 and then falls, while the peak grows without
visible bound (40 at lam = 0.455, in agreement with arXiv:2511.22819, which reports "nearly 40" there and a
persisting non-smooth signal on [0.455, 0.4713]). On the last nine points mu - 2 is linear in 1/peak with
intercept +0.0022 at infinite peak, reached near lam = 0.4535 to 0.4545. If that extrapolation (over a factor
1.7 in peak) holds, this family ends in a singular profile before becoming smooth again, and a third smooth
profile, if it exists, is not on it. We could not continue past lam = 0.454544.

## 7. What is not claimed

- No proof of anything. No stability analysis of the profiles (the words "first/second unstable" follow the
  literature's labelling at a = -1 and are carried along the branch by continuity; the number of unstable
  directions along the a-branch was not computed). [TODO]
- Novelty of items 2 and 4 rests on a literature search by automated agents plus a direct reading of three
  papers (arXiv:2509.14185, 2511.22819, 2603.25104). [TODO: a person should check, in particular
  Lushnikov-Silantyev-Siegel and later work on gCLM branches.]
- The third smooth CCF profile: not found.

## 8. Reproduction

Windows, Python venv with torch 2.6 (CUDA), no other dependencies; `export PYTHONPATH=.` in the repo root.
Commands to resume or re-run each computation are in `STATE.md` section 4; the scripts are listed in
`runs/README.md`. Solutions are stored in `runs/*.pt`, one-line-per-solve ledgers in `runs/*.jsonl`.

## References (to be completed)

- A. Cordoba, D. Cordoba, M. Fontelos, Formation of singularities for a transport equation with nonlocal
  velocity, Ann. of Math. 2005.
- H. Okamoto, T. Sakajo, M. Wunsch, On a generalization of the Constantin-Lax-Majda equation, Nonlinearity 2008.
- P. Lushnikov, D. Silantyev, M. Siegel, Collapse versus blow-up and global existence in the generalized
  Constantin-Lax-Majda equation, J. Nonlinear Sci. 2021.
- Y. Wang et al., Discovery of unstable singularities, arXiv:2509.14185.
- Resolving sharp gradients of unstable singularities to machine precision via neural networks, arXiv:2511.22819.
- D. Huang, J. Tong, X. Wang, Self-similar finite-time blowups with singular profiles of the generalized
  Constantin-Lax-Majda model, arXiv:2603.25104.
