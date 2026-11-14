# HANDOFF â€” gpu-solver branch (builder session on Bobbo, RTX 4080 SUPER)

Written by the builder (Opus) session. The math session answers in the
"Answers from the math session" section at the bottom. Every number below was
run on this machine; the full records are in `runs/ledger.jsonl` (one JSON line per run).

## Machine (inventory 2026-11-11)
- GPU: NVIDIA GeForce RTX 4080 SUPER, 16 GB (about 2.5 GB used by the desktop at idle), driver 610.88, CUDA UMD 13.3.
- Python: 3.11.0 (system, no torch), 3.13.13 (miniconda: torch 2.6.0+cu124, CUDA available), 3.10/3.12 under StabilityMatrix.
- JAX: not installed. scipy and pytest: not available in the venv (nothing installed).
- Disk C: about 356 GB free.
- Ollama installed. LM Studio installed (X:\LLAMA, currently running).
- Environment: `.venv/` in the repo root, created from miniconda Python with `--system-site-packages`
  (reuses the existing torch; nothing installed system-wide or into the venv).

## What was built
`gpu_solver/` â€” torch, float64, on CUDA.
- `lm.py`: Levenbergâ€“Marquardt, Jacobians from `torch.func.jacrev`. The damped step is solved by QR of the
  column-scaled augmented Jacobian. (Normal equations were the first version; they square the condition number.
  Those runs are kept in `runs/ledger_attempt1_normal_eqs.jsonl`.)
- `burgers.py` (1a): ansatz `U = -y s N(s)`, `s = (1+y^2)^(-a/2)`, `a = 1/(1+lam)`,
  `N(s) = 1 + s(1-s) * sum_k c_k T_k(2s-1)`. So U is odd and analytic at y = 0.
  `N(1) = 1` fixes U'(0) = -1, and `N(0) = 1` fixes the far field U ~ -y^(lam/(1+lam)), i.e. C = 1 in
  y = -U - C U^(1+1/lam). The far-field factor excludes the trivial solution U = -y. lam is an unknown.
  U'''(0) = 3a(1 - P(1)), so U'''(0) = 0 is equivalent to sum_k c_k = 1 (one extra residual row).
  Optional rows: the residual at complex points y = x + i*{0.1, 0.2}, x in [0.05, 4]. The exact profiles'
  nearest branch points are at |Im y| â‰ˆ 0.385 (lam = 1/2) and â‰ˆ 0.38 (lam = 1/4).
- `clm.py` (1b): y = tan(Î¸/2), W = sum_{k<=K} b_k sin kÎ¸, y W' = sin Î¸ Â· F'(Î¸),
  H W = -sum b_k cos kÎ¸ + sum b_k (-1)^k (Cayley map). Normalisation W'(0) = -2. c is fixed at 1 or left free.
- `hilbert.py`: FFT Hilbert transform on R via the Cayley map.
- `tests/test_gpu_solver.py`: 4 checks, all pass (three closed-form Hilbert pairs to < 1e-12; closed form vs FFT;
  the exact profiles have zero residual).

## Results (rung 1)

### 1b CLM â€” PASS
Reference: W = -2y/(1+y^2), c = 1 (exact, DIRECTION.md).

| run | K modes | seeds | c found | sup residual, \|y\|<=50 | sup profile error, \|y\|<=50 |
|---|---|---|---|---|---|
| c fixed = 1 | 16, 32 | 0, 1 | 1 | <= 8.9e-16 | <= 1.2e-15 |
| c free, c0 = 0.8 and 1.3 | 16, 32 | 0, 1 | 1 - 7.1e-14 (worst) | <= 5.5e-15 | <= 1.6e-13 |

The pass condition (residual < 1e-8, profile error < 1e-6) holds in all 12 runs. With c free, c = 1 is recovered.
Caveat: the exact profile is a single sine mode in this basis, so this tests the nonlocal operator and the
Newton machinery more than the resolution.

### 1a Burgers, lam free (stable branch) â€” PASS
Reference: lam = 1/2, y = -U - U^3 (exact).

| n basis | colloc | lam0 | lam found | \|lam - 1/2\| | sup profile err, \|y\|<=20 | sup residual |
|---|---|---|---|---|---|---|
| 24 | 48 | 0.45, 0.55, 0.70 | 0.4999358460 | 6.4e-5 | 2.0e-4 | 2.3e-7 (basis too small: fails) |
| 48 | 96 | 0.45, 0.55, 0.70 | 0.4999999974 | 2.6e-9 | 8.2e-9 | 2.8e-12 |
| 96 | 192 | 0.55 | 0.500000000000 | 2.5e-13 | 7.7e-13 | 1.1e-15 |
| 192 | 384 | 0.55 | 0.500000000001 | 1.4e-12 | 4.4e-12 | 1.8e-15 |
| 96, + complex rows | 192 | 0.55 | 0.500000000000 | 1.2e-14 | 4.1e-14 | 1.3e-15 (complex 2.6e-15) |

Two seeds each, identical to the digits shown. Wall time 0.2â€“2 s per solve. lam is stable under doubling the
basis (48 â†’ 96 â†’ 192).

### 1a Burgers with U'''(0) = 0 (unstable branch, lam = 1/4) â€” FAIL (threshold 1e-6)
Reference: lam = 1/4, y = -U - U^5 (exact).

| n | rows | lam0 | lam found | \|lam - 1/4\| | profile err vs y = -U - U^5 |
|---|---|---|---|---|---|
| 24 | real | 0.2, 0.3 | 0.2501506 | 1.5e-4 | 4.9e-4 |
| 48 | real | 0.3 | 0.2499963332 | **3.7e-6 (best)** | 1.2e-5 |
| 48 | real + complex | 0.3 | 0.2500043433 | 4.3e-6 | 1.4e-5 |
| 96 | real | 0.3 | 0.2561 / 0.2580 (seed 0/1) | 6â€“8e-3 | 2â€“3e-2 |
| 192 | real | 0.3 | 0.2707 / 0.2757 | 2â€“3e-2 | 7â€“8e-2 |
| 96 | real + complex | 0.3 | 0.2841 / 0.2697 | 2â€“3e-2 | 7e-2 â€“ 1e-1 |

The error gets **worse** as the basis is refined. This is not convergence.

Diagnosis, partly measured (`runs/representation_check.txt`): with lam fixed at 1/4, the basis fits the exact
profile to 4.2e-11 (n = 48) and 4.5e-13 (n = 96). Representation is therefore not the problem; the residual
pins lam down only weakly. For lam â‰  1/(2i) the exact sublinear solutions contain |U|^(1+1/lam), which is
C^floor(1/lam) at y = 0. Near lam = 1/4 that term is about C^4, and a larger polynomial basis imitates it with a
residual (about 1e-9 to 5e-10) no larger than the true profile's at that basis size, so the minimiser drifts in lam.
Near lam = 1/2 the competing term is only about C^2 and is ruled out sharply, which is why the stable branch passes to
1e-13. The complex off-axis rows did not fix it at n = 96. The C^4 explanation is my reading; I have not proved it.

Other observations (recorded, not targets):
- lam free, lam0 = 0.40 lands on the lam = 1/4 branch (n = 48 with complex rows: 0.2499959, profile error 1.4e-5
  against y = -U - U^5). That is consistent: U'''(0) = 0 holds automatically on that branch.
- U'''(0) = 0 with lam0 = 0.20 drifts toward 1/6 (0.178 at n = 48) or 1/8 (0.142 at n = 96) without converging.
  U'''(0) = 0 holds on every branch i >= 2, so the constraint selects the 1/4 branch only from nearby starts.

## Done / not passing / skipped / unverified
- Done: inventory; clone; branch `gpu-solver` from `origin/direction`; 1b PASS; 1a stable branch PASS;
  operator tests; run ledger.
- Not passing: 1a unstable branch (lam = 1/4 to 1e-6). Best 3.7e-6.
- Skipped: rung 2 (gated: 1a is not fully passed, and Michael has not said go). JAX not tried.
- Unverified: whether rows for the derivatives of the residual fix lam = 1/4. The attempt (`runs/smooth_trial.py`,
  nested forward-mode derivatives) was too slow (no solve finished in 13 minutes) and was stopped. No result.
- Not used: Sonnet or Haiku subagents. The work was small enough to do inline.

## Questions for the math session
1. Must the 1a unstable branch pass before rung 2? The stable branch passes to 1e-13, and the failure is
   diagnosed as lam being weakly determined, not as poor representation. If it must pass, which fix?
   (a) Impose smoothness at the origin directly: compute the Taylor series at y = 0 by exact recursion and match it
       to the global basis at some y0, so that the |U|^(1+1/lam) term cannot appear.
   (b) Penalise spectral-tail coefficients (Tikhonov, weights growing in k).
   (c) Choose n at the residual-versus-n plateau instead of refining further, and report lam versus n.
   My lean is (a): it carries over to rung 3, where smoothness at the origin is what selects the profile.
2. The same weakness is likely in rungs 2 and 3: when the competing non-smooth family is smooth to high order,
   residual minimisation pins lam only weakly. For CCF and for Boussinesq with boundary, how smooth is the nearest
   non-smooth competing family? That decides how hard lam is to pin down and which smoothness rows to add.
3. Rung 2a (CCF): please confirm the target exponents in arXiv:2509.14185 (table or equation number). I have not read it.

## Answers from the math session
Written 2026-11-11 by the math session, against commit f06e93f. Derivations marked "checked" were
verified symbolically in that session; statements marked "inference" were not.

### Verdict on rung 1
1b passes. 1a stable passes. 1a unstable fails the stated threshold, and your diagnosis is right. I
agree with stopping at the gate.

### Q1. Fix the unstable branch with origin Taylor rows, then it must pass

Why residual minimisation cannot pin lam here (checked). Your ansatz is analytic at y = 0 for every
parameter value. Write U = -y + u3 y^3 + u5 y^5 + u7 y^7 + ... and expand the residual
R = -lam U + ((1+lam) y + U) U' at the origin:

    r1 = 0
    r3 = u3 (2 lam - 1)
    r5 = u5 (4 lam - 1) + 3 u3^2
    r7 = u7 (6 lam - 1) + 8 u3 u5

So an analytic solution other than U = -y exists only at lam = 1/(2i): the first nonzero coefficient
u_(2i+1) is allowed only when (2i lam - 1) = 0. The exponent is fixed by an order-one algebraic
condition at the origin. Collocation away from the origin sees a wrong lam only through the error of
approximating |U|^(1+1/lam) by the basis, which is the small number E_n you measured (about 1e-9).
The residual as a function of lam is therefore a V with slope about E_n. E_n shrinks as n grows, so
the V flattens and lam is determined to about (solver noise)/E_n, which gets worse with refinement.
That is the behaviour in your table, and it is why n = 48 was best.

The fix is your option (a), in this specific form:
- Compute the Taylor coefficients of the ansatz at y = 0 by truncated power-series arithmetic in
  torch (series in y^2, multiply and compose as polynomials). This is exact, cheap and
  differentiable. Do not use nested autodiff.
- Form r3, r5, r7 from those coefficients and add them as residual rows with weight of order one
  (not scaled down to the size of the collocation residual).
- For the lam = 1/4 branch keep the row u3 = 0. Then r5 = u5 (4 lam - 1), and u5 is not zero because
  your far-field normalisation excludes U = -y. For the exact profile y = -U - U^5, u5 = 1.
- Do not use (b). A tail penalty biases the profile and does not add the missing information.
  (c) is a correct description of the trade-off but caps accuracy near 1e-6.

Pass condition, unchanged: |lam - 1/4| < 1e-6, and the error must not grow when n doubles
(48 -> 96 -> 192). I expect far better than 1e-6; if the error still grows with n, stop and report.

One cheap diagnostic to record first: with lam fixed on a grid of about 21 values in [0.24, 0.26],
solve for the coefficients only and log the minimum residual norm against lam at n = 48 and n = 96.
The prediction is a V centred at 0.25 whose slope drops as n rises. If it is not a V, the diagnosis
above is wrong and I need to see it.

The gate: 1a unstable must pass before rung 3 and before any unstable target in rung 2. The stable
targets of rung 2 (2a stable, 2b stable) do not depend on it and may start once Michael says go,
but build the origin rows first: it is a few hours of work and rung 2 needs them anyway.

### Q2. How smooth is the competing family in rungs 2 and 3

The general mechanism (inference from the Burgers case, standard for transport-type profile
equations). At a stagnation point of the self-similar flow the transport operator is of Euler type,
(c + U'(0)) y d/dy plus a constant, so the linearised equation has homogeneous solutions |y|^mu with
mu a continuous function of lam and of the origin gradient of the velocity profile. A smooth profile
needs mu to be an integer of the right parity. The competing non-smooth family is C^floor(mu). The
n-th admissible lam is where mu reaches the n-th allowed integer, so:
- the stable profile sits at the lowest resonance and has the least smooth competitor, as with
  Burgers lam = 1/2. It should be sharply determined by collocation alone;
- each further unstable profile has a competitor about two orders smoother, so each is harder to
  pin by the same factor you saw between lam = 1/2 and lam = 1/4.

Supporting evidence, not proof: arXiv:2509.14185 reports that 1/lam is close to linear in the
instability order n for IPM, and 1/(lam - 1) for Boussinesq (their fitted rules), which is the
pattern mu(lam) crossing successive integers would give. Burgers has 1/lam = 2i exactly.

I have not done the local analysis for CCF or Boussinesq, so I cannot give you the integers. What
to do instead, for every profile you converge:
1. Report the origin exponent. Linearise the profile equation at the stagnation point using your
   computed U'(0) (for Boussinesq, the two velocity-gradient eigenvalues at the corner) and record
   mu. This number tells us which resonance the profile sits on and how smooth the competitor is.
2. Add origin Taylor rows through order ceil(mu) + 1, built the same way as for Burgers. Nonlocal
   terms enter those rows only through the Taylor coefficients of the velocity, which the basis
   gives in closed form.
3. Prefer an analytic relation for lam from the origin when one exists, with lam eliminated as an
   unknown. arXiv:2509.14185 says (Methods, "Incorporating mathematical structure") that
   "smoothness requirements at the origin can also be used to derive an analytical relationship
   for the self-similar parameter lambda" and treats this as more robust than training lam. A
   summary of that paper gives the Boussinesq relation as lam = -3 - 2 dU1/dy1 at the origin; I
   could not verify that formula against the text, so derive it yourself before using it.

For rung 3 this matters more, not less: there is no reference value, so lam must come from an
origin condition, with collocation used only to solve for the profile.

### Q3. CCF targets in arXiv:2509.14185

What I could confirm from the page text (Introduction):
- first unstable CCF profile: lam_1 â‰ˆ 0.6057 (originally Wang et al., recovered by Eggers and Fontelos);
- second unstable CCF profile: lam_2 = 0.4703, found in that paper, residual of order 1e-7;
- fitted rules: Boussinesq lam_n ~ 1/(1.4187 n + 1.0863) + 1, IPM lam_n ~ 1/(1.1459 n + 0.9723).

What I could not confirm: the stable CCF exponent lam_0 (shown only in Figure 2f, which I could not
read), the CCF equation and sign conventions, the profile equation, the normalisation, and all exact
Boussinesq values. The paper's Supplementary Information was not posted with v1. No table numbers
could be confirmed.

So for 2a: read the equation, ansatz and lam_0 from the PDF of arXiv:2509.14185 and from
arXiv:2201.06780, and write them at the top of the 2a section of this file with page references
before running. The target is lam_0 to 4 digits and lam_1 = 0.6057 to 4 digits. Check your sign and
scaling conventions against lam_1 first, since that is the one value I can vouch for.

### Order of work
1. lam-scan diagnostic (minutes).
2. Origin Taylor rows; rerun 1a unstable; report against the pass condition.
3. Wait for Michael's go on rung 2.


## Builder update 2026-11-13 — rung 1 now passes in full (run on this machine, RTX 4080 SUPER)

Followed the math session's order of work. Records: `runs/lam_scan.jsonl`, `runs/ledger.jsonl`
(cases `1a-burgers-origin*`), raw output in `runs/origin_continuation.out`.

### 1. lam-scan diagnostic — prediction confirmed
lam fixed on 21 values in [0.24, 0.26], coefficients solved (old rows: collocation + sum c = 1), all 21
in one batched solve (3.5 s per n).

| n | min \|r\|_2 (at lam = 0.250) | \|r\|_2 at 0.249 / 0.251 | \|r\|_2 at 0.240 / 0.260 | slope d\|r\|/dlam |
|---|---|---|---|---|
| 48 | 3.0e-9 | 7.0e-8 / 7.8e-8 | 4.4e-7 / 1.2e-6 | about 7e-5 |
| 96 | 1.1e-10 | 4.8e-9 / 5.4e-9 | 2.7e-8 / 8.9e-8 | about 5e-6 |

A V centred at 0.25 whose slope drops about 14x from n = 48 to n = 96. (The V is not symmetric.)

### 2. Origin Taylor rows — 1a unstable PASS, 1a stable still PASS
Built as prescribed: `burgers.origin_series` gets the Taylor coefficients of the ansatz at y = 0 by
truncated power-series arithmetic (closed-form Chebyshev table T_k(1+2x) = sum_j d[k,j] x^j, binomial series
for s, polynomial products), `origin_rows` forms r3, r5, r7 from the series of the residual, weight 1.
A test checks them against your hand formulas (agree to 1e-15 relative). The u3 = 0 row is taken from the series.

Result, 16 starts (lam0 in {0.20, ..., 0.36} x 2 seeds), solved as one batch, then continued in n:

| branch | n | starts passing | \|lam - ref\| (worst) | sup profile err, \|y\|<=20 | sup residual | batch wall |
|---|---|---|---|---|---|---|
| u3 = 0 (ref 1/4) | 48 | 16/16 | 7.4e-15 | 5.6e-9 | 5.3e-10 | 8.1 s |
| u3 = 0 | 96 | 16/16 | 1.4e-16 | 2.0e-10 | 1.1e-11 | 9.5 s |
| u3 = 0 | 192 | 16/16 | 1.9e-16 | 2.0e-10 | 1.1e-11 | 2.9 s |
| lam free (ref 1/2) | 48 | 10/12 | 2.0e-13 | 1.5e-11 | 8.9e-12 | 7.8 s |
| lam free | 96 | 10/12 | 1.7e-16 | 5.6e-14 | 3.6e-15 | 7.0 s |
| lam free | 192 | 10/12 | 2.8e-16 | 5.6e-14 | 3.3e-15 | 1.3 s |

Pass condition (|lam - 1/4| < 1e-6, no growth under 48 -> 96 -> 192): met, by nine orders of magnitude.
u5 = 1.000000 on the 1/4 branch (exact value 1) and u5 = -3.000000 on the 1/2 branch.

Things you should know before trusting this further:
- **Initialisation matters now.** With white-noise initial coefficients (the old default) the origin rows made
  convergence much worse: 7/16, 3/16 and 0/16 starts converged at n = 48, 96, 192 (`runs/origin_trial.out`,
  ledger case `1a-burgers-origin-U3zero`), because the rows weigh c_k by about k^6. What works: initial noise
  decaying like k^-3, multi-start at n = 48, then pad with zeros and polish at n = 96 and 192
  (`runs/origin_continuation.py`). The n = 192 rows above are therefore polish steps (1-3 iterations), not
  independent solves; the profile error does not improve from 96 to 192.
- **The two lam-free "non-passes" are not failures.** Starts at lam0 = 0.80 converged to the lam = 1/4 branch
  (lam = 0.25 to 1e-16, u3 = 0 automatically, residual 1e-11). It is a true solution, just not the reference
  for that row. With origin rows the solver finds whichever branch the start is closest to in its own metric.
- In the failed white-noise runs, unconverged starts drifted toward u5 = 0 and lam between 0.05 and 0.19,
  i.e. toward the higher branches 1/6, 1/8, ..., which satisfy all imposed rows. None converged in 400 iterations.
  Reported u3, u5, u7 identify the branch.
- Not done: no analytic elimination of lam (lam stayed an unknown, as DIRECTION.md 1a requires).

### 3. Batched multi-start (Michael asked for it)
`gpu_solver/batch_lm.py`: B independent LM problems in one set of batched QR solves, per-problem damping,
chunked Jacobians (the first version ran out of the card's 16 GB at n = 192; chunk = 64 fixed it), stall stop.
16 starts cost about the same wall time as one.

### 4. Literature for 2a (Sonnet subagent, NOT run or checked by me) — `runs/lit_2a_notes.md`
Reported as read from arXiv:2509.14185 v1 Fig. 2f and arXiv:2201.06780: CCF stable lam_0 = 1.1807776628998
(2201.06780: 1.18078), first unstable 0.6057337012032, second 0.4713248638620; Boussinesq stable
1.9205599746927; Boussinesq relation lam = -3 - 2 d_y1 U1(0,0) quoted from Methods.
Discrepancy for you: the text of 2509.14185 says lam_2 = 0.4703, its Fig. 2f says 0.4713. Unresolved.
Not found: the sign and prefactor of the Hilbert transform in the CCF papers, and the CCF equations in
2509.14185 (Supplementary not posted). The 2201.06780 equations were partly reconstructed from a text
extraction that dropped Greek letters, so I will re-derive the profile equation and check it against
lam_1 = 0.6057 before trusting any of it. The agent also offered its own origin relation for CCF
(lam = -2 + 2U'(0)); unverified, please check.

### State
- Rung 1: 1a stable PASS, 1a unstable PASS, 1b PASS.
- Rung 2: not started. Waiting for Michael's go.
- Question: for CCF, is the stable profile smooth at the origin with a C^k competitor as in your Q2 picture,
  and is lam = -2 + 2U'(0) (convention omega_t - u omega_x = u_x omega, Omega'(0) = -2) right?


## Builder update 2026-11-13 (later) — rung 2a: stable CCF PASS to 13 digits; unstable NOT reproduced

Michael said go on rung 2 ("Go with your instincts. Make serious contributions.") after I proposed aiming
2a at the unstable profiles. Everything below was run on this machine unless marked "subagent".

### Equation, ansatz, conventions (derived here; consistent with the subagent's reading of arXiv:2201.06780)
    omega_t + a u omega_x = u_x omega,  u_x = H omega,  H f(y) = (1/pi) p.v. int f(y')/(y-y') dy',  a = -1 (CCF)
    omega = (T-t)^-1 Omega(y),  y = x/(T-t)^(1+lam)
    Omega + ((1+lam) y + a U) Omega' - Omega U' = 0,  U' = H Omega,  U(0) = 0,  Omega odd,  Omega'(0) = -2.
Convention check: a = 0, lam = 0 is the exact CLM profile of rung 1b.
First Taylor order at y = 0 gives lam = (1 - a) U'(0) - 2, so lam = 2 U'(0) - 2 for CCF (the subagent's
relation is right). For a = -1 the equation integrates once to the Burgers form
    -lam Theta + ((1+lam) y - H Theta) Theta' = 0,   Theta' = Omega, Theta even, Theta(0) = 0.

### Method (`gpu_solver/ccf.py`)
- Unknowns: lam and c_k in Theta = r^beta M(s), r = 1+y^2, alpha = 1/(1+lam), beta = (1-alpha)/2,
  s = r^(-alpha/2), M = sum c_k T_k(2s-1). Even, analytic at 0, tail Omega ~ y^(-alpha) with corrections in
  powers of s (an assumption; spectral decay of c_k for the stable profile supports it there).
- U = H Theta and U' = H Omega by trapezoid quadrature on y' = sinh(j h), evaluated at the midpoints
  (a pole midway between nodes gives the principal value with exponentially small error), parity used
  explicitly, tail beyond sinh(L) added in closed form. Check against the exact CLM pair: errors <= 5e-13
  (`runs/ccf_operator_check.py`).
- Rows: collocation at the midpoints up to xi = Lc, Omega'(0) = -2, and lam + 2 - 2 U'(0) = 0. Batched LM.

### Stable profile — PASS (target: 4 digits)
Reference 1.1807776628998 (arXiv:2509.14185 v1 Fig. 2f, read by subagent); 1.18078 in arXiv:2201.06780.

| quad step h | L / Lc | n | lam | sup residual |
|---|---|---|---|---|
| 0.2 | 40 / 22 | 72 | 1.18077944560844 | 7e-13 |
| 0.1 | 40 / 22 | 48, 72, 96 | 1.18077767678473 | 2e-13 |
| 0.05 | 40 / 22 | 96 | 1.18077766309102 | 3e-14 |
| 0.025 | 40 / 22 | 96 | 1.18077766290000 | 9e-14 |
| 0.025 | 55 / 28 | 96 | 1.18077766289987 | 4e-14 |
| 0.0125 | 40 / 22 | 96 | 1.18077766289987 | 3e-14 |

lam_0 = 1.1807776628999 (last digit uncertain), U'(0) = 1.590388831450. Agreement with the published value
to 13 significant digits, lam solved as an unknown. All 12 starts lam0 in [0.3, 2.2] at n = 24 converge to it.
Warning recorded: the residual is NOT a reliable error indicator here. At h = 0.1 the residual was 2e-13
while lam was wrong by 1.4e-8, because the quadrature error is a consistent perturbation of the operator.
Only the h-refinement exposed it. The lam scan at fixed lam (n = 32) shows an order-one V at 1.18: the
stable exponent is sharply determined, as you predicted in Q2.

### Unstable profiles — NOT reproduced (negative result, please read)
- With lam free, starts below about 0.75 do not converge. They creep along a valley: lam 0.60 -> 0.499
  (300 its) -> 0.4526 (600 more), sup residual 7e-5 -> 1e-5, passing 0.6057 and 0.4713 without stopping.
- lam fixed at the published 0.6057337012032 and at 0.6040, 0.6075, 0.5900, 600 iterations each, n = 48
  (`runs/ccf_scan_fixed48.out`): residual l2 6.2e-5, 5.9e-5, 6.6e-5, 3.8e-5. No dip at the published value;
  the residual just falls as lam falls. None converged; trailing coefficients about 1e-4, |c_k| flat near
  1e-3 for k = 12..32 (the stable profile's reach 1e-11 by k = 44).
- So in this basin there is a continuum of near-solutions that are not smooth in s, and the smooth unstable
  profile is either in another basin or not distinguished at n = 48. This is the Burgers 1/4 situation again,
  but I have not found the selecting condition:
  - The origin gives nothing new. With lam = 2U'(0) - 2 imposed, order y^(2j) of the Burgers form reads
    theta_j (j-1) lam = sum_{i=1}^{j-1} 2 (j-i) u_i theta_(j-i): no resonance, every order is solvable. A
    local term B|y|^nu at the origin is excluded at leading order unless nu = 2. (Derived here, unchecked.)
  - Transport speed (1+lam) - U/y stays positive on the valley solutions (0.30 at the origin, 1.57 at
    infinity), so there is no interior stagnation point.
  - That leaves the far field (a tail power off the s-lattice) or complex singularities near the real axis.
- Also: the solver is slow in the valley (0.7 s per iteration, hundreds of iterations). At fixed lam the
  residual is quadratic in c, which I have not exploited.

### Prior art (Sonnet subagent, `runs/lit_prior_art_ccf.md`; not verified by me)
- 2509.14185 has only v1; "0.4703" in its text is a typo for the Fig. 2f value 0.4713248638620. A follow-up,
  arXiv:2511.22819, gives lam_2 = 0.47132422 (differs at the 7th digit) and found no third profile in
  [0.455, 0.4713]. arXiv:2610.30492 (ODIL, Chebyshev + Gauss-Newton) reproduces the three profiles with
  the published lam values. So "resolve the typo" is no longer a contribution.
- No rigorous or computer-assisted proof of an unstable profile was found for any of these models.
- Open and reachable: an independent lam_2 beyond 6 digits; a third CCF profile; a rigorous proof.

### Questions for the math session
1. What selects the smooth unstable CCF profiles? Where is the general solution of
   -lam Theta + ((1+lam) y - H Theta) Theta' = 0 (lam below lam_0, Theta ~ theta_1 y^2 at 0) non-smooth:
   at infinity (which tail exponents are allowed beyond y^(lam/(1+lam)) times powers of y^(-1/(1+lam))?),
   or through complex singularities? I need the row to add, as with r5 for Burgers.
2. Is the unstable profile perhaps not of the form Theta ~ y^2 at the origin (e.g. Omega'(0) = 0)?
   arXiv:2201.06780 pins the unstable solution with a condition at y = 0.5 instead of Omega'(0) = -2.
3. For the stable profile the trapezoid error behaves like a strip half-width of about 0.15 in xi = asinh(y).
   Does that match a known complex singularity of the profile?


## Builder update 2026-11-13 (third) — corrections, unstable search still negative, gCLM stable branch

### Corrections to the previous update
1. **The "valley" of near-solutions below lam = 0.61 was an artifact, not a family.** At quadrature step
   h = 0.1 the collocation points (the trapezoid midpoints) are too sparse in the interior of s in (0,1) to
   determine a degree-48 polynomial, so the least-squares problem has near-null directions and the residual
   is small for functions that are not solutions. With h = 0.05 the same coefficient vectors have residual
   l2 about 0.47 (`runs/ccf_fixed.py`). Everything said about a valley, and question 1 as posed, is withdrawn.
2. **Same cause for the stable profile's "h-dependence".** At h = 0.1, n = 96, starting from the h = 0.025
   solution, LM stops at lam = 1.18077766293581 with residual 2e-13; starting from scratch it had stopped at
   1.18077767678473 with residual 2e-13. So at h = 0.1 a set of lam values 1.4e-8 wide all pass the residual
   test: under-determination, not quadrature error. My "strip half-width 0.15" inference (question 3) is
   withdrawn. The h = 0.025 and 0.0125 results stand (rows 880 and 1760 for 97 unknowns):
   lam_0 = 1.1807776628999.
3. New, faster solver `gpu_solver/ccf_fast.py`: for fixed lam all fields are linear in c, so the Jacobian is
   explicit; the lam column is a central difference. About 0.1 s per iteration. On this problem size the CPU
   is 4x faster than the GPU (tiny tensors), so these runs used the CPU. Old and new residuals agree to 6e-15.
4. A second representation, `gpu_solver/ccf_family.py`: Theta = r^beta (1-s)^kappa N(s), kappa = (nu+1)/2,
   which builds in Omega ~ |y|^nu at the origin. At nu = 1, n = 32, h = 0.05 it returns
   lam_0 = 1.180777662896 in 13 iterations (1 s) from the trivial start N = -1.

### Unstable CCF profiles: five attempts, all negative
Local analysis used (derived here, not checked by anyone else): for a = -1 and analytic odd U,
Theta = exp(int lam dy / ((1+lam) y - U)) so Theta ~ |y|^mu at 0 with mu = lam / (1 + lam - U'(0));
equivalently Omega ~ |y|^nu, nu = mu - 1 = (U'(0) - 1)/(1 + lam - U'(0)). Smooth iff mu is an even integer.
For CLM (a = 0) the same formula reproduces the exact family g = 2/(1 + C y^(1/c)) (nu = 1/c).

| attempt | set-up | outcome |
|---|---|---|
| A | mu = 2 (Omega'(0) = -2, lam = 2U'(0) - 2), lam fixed at the published 0.6057337012032, n = 24, h = 0.05, 60 random starts | all end in 3-4 local minima, residual l2 0.65 - 0.92. No solution. |
| B | as A but no origin conditions (tail normalisation only), 30 starts | best l2 0.39; the same set-up at lam_0 also fails to find the stable profile (best 4e-2), so this test is inconclusive |
| C | mu = 4 (Omega ~ y^3) via constraint rows, h = 0.1 (aliased) | not valid evidence; superseded by D |
| D | mu = 4 built in exactly (`ccf_family`, nu = 3), lam free, n = 24, h = 0.05, 60 random starts lam0 in [0.35, 1.25] | 57 of 60 drift to lam near -0.94 (degenerate, alpha -> infinity, not converged); 2 stop at lam = 0.8094 with residual 1.0. No solution near 0.6057. |
| E | continuation in nu from 1 to 3.2 in steps of 0.1 | residual grows immediately (4e-3 at nu = 1.1); expected, because for non-odd nu the origin expansion has powers |y|^(nu + 2i + (nu-1) j) that the basis cannot represent; uninformative |
| F | Picard iteration on the characteristic form to map mu(lam) (`runs/ccf_picard.py`) | does not converge even at lam_0; uninformative |

So: I cannot find the published unstable profiles with either mu = 2 or mu = 4 at the origin and a tail on the
s-lattice. Either the profiles violate one of those two structural assumptions, or the basin is small and my
starts miss it. I do not know which. What would settle it quickly: the profile data or the SI of
arXiv:2509.14185 / 2201.06780 (value of U'(0) for the first unstable profile: mu = 2 needs 1.30287, mu = 4
needs 1.45430), or Fig. 3 of 2201.06780 read by a person.

### gCLM stable branch (continuation in a from the CCF profile)
Model omega_t + a u omega_x = u_x omega; lam solved as an unknown with the row lam = (1-a) U'(0) - 2.
"A" = n 80, h 0.025 (n 96 for a < -1); "B" = n 64, h 0.05. Raw output: `runs/gclm_up80.out`,
`runs/gclm_down96.out`, `runs/gclm_up64.out`, `runs/gclm_branch.jsonl`.

| a | lam (run A) | sup residual A | \|A - B\| |
|---|---|---|---|
| -2.0 | 2.3339891442745 | 5.5e-7 | not run |
| -1.5 | 1.7586510750825 | 1.9e-9 | not run |
| -1.2 | 1.4123386442351 | 1.8e-12 | not run |
| -1.1 | 1.2966402821692 | 2.6e-13 | not run |
| -1.0 | 1.1807776628986 | 1.4e-12 | 4.3e-10 |
| -0.9 | 1.0647228941224 | 9.2e-14 | 4.2e-11 |
| -0.8 | 0.9484407856524 | 4.3e-14 | 3.0e-12 |
| -0.7 | 0.8318864440647 | 8.2e-14 | 8e-13 |
| -0.6 | 0.7150018957359 | 4.1e-13 | 3.0e-12 |
| -0.5 | 0.5977112640435 | 1.3e-12 | 9e-13 |
| -0.4 | 0.4799137503026 | 3.8e-13 | < 1e-13 |
| -0.3 | 0.3614732023901 | 2.6e-11 | 4.7e-12 |
| -0.2 | 0.2422022487452 | 1.9e-10 | 1.1e-10 |
| -0.1 | 0.1218375392812 | 6.6e-10 | 2.2e-8 |
| -0.05 | 0.0611326351965 | 7.2e-10 | 5.8e-6 |

Reading: for -1 <= a <= -0.3 two resolutions agree to between 1e-10 and 1e-13, so about 10 digits are
supported. Near a = 0 the representation degrades (the exact CLM profile has Theta = -ln(1+y^2), a logarithm
the basis cannot hold) and only the trend lam -> 0 is confirmed. For a < -1 there is one resolution only and
the residual grows; treat those as provisional. The only external check is a = -1 (13 digits). I know of no
published high-precision lam(a) to compare with (Lushnikov-Silantyev-Siegel's fit at a = -1 is 1.2% off per
the subagent's notes); this table is unverified outside its own refinement.



## Builder update 2026-11-13 (fourth) — first unstable CCF profile found (6 digits)

What unblocked it: I opened Fig. 3 of arXiv:2201.06780 (ar5iv, `CCF_compare.png`) in the browser and read it
myself. The unstable profile there has U'(0) about 1.30 (so mu = 2: lam = 2U'(0) - 2 holds, as for the stable
profile; my mu = 4 hypothesis is wrong), Omega'(0) about -0.1 in their normalisation, and a sharp peak
|Omega| about 1.6 near y = 4. In my normalisation Omega'(0) = -2 that peak sits at y about 0.2, which the
basis could not resolve and no start resembled. The figure was used only to shape the initial guess and to
choose the normalisation Omega'(0) = -0.1; lam was then solved as an unknown.

Set-up (`runs/ccf_unstable1.py`, `runs/ccf_u_refine.py`, solver `gpu_solver/ccf_fast.py`): same equations and
rows as the stable case, normalisation Omega'(0) = -0.1, initial guess
Omega = -(0.1 y + k y^3)/(1 + (y/4)^8)^((3+alpha)/8) fitted to the basis, lam fixed for 400 iterations and then free.

| n | quad step h (rows) | lam | sup residual | U'(0) | peak \|Omega\| at y | note |
|---|---|---|---|---|---|---|
| 48 | 0.05 (440) | 0.6598858 | 2.0e-2 | 1.33999 | 1.254 at 2.72 | |
| 64 | 0.05 | 0.6337437 | 9.3e-3 | 1.32152 | 1.388 at 2.58 | |
| 128 | 0.05 | 0.6109922 | 7.2e-13 | 1.30550 | 1.546 at 2.44 | **aliased: not a solution** (too few rows for n) |
| 128 | 0.025 (880) | 0.6076754 | 7.8e-4 | 1.30413 | 1.572 at 3.49 | |
| 128 | 0.0125 (1760) | 0.6085334 | 1.1e-3 | 1.30479 | 1.567 at 3.56 | |
| 192 | 0.0125 | 0.60573472576 | 6.5e-6 | 1.3028675527 | 1.58984 at 3.746 | GPU |
| 288 | 0.0125 | 0.60573467431 | 3.0e-9 | 1.3028673372 | 1.58985 at 3.746 | GPU; stopped at the 300-iteration cap |

Published lam_1 = 0.6057337012032 (2509.14185 Fig. 2f), 0.60573 (2201.06780). Mine: 0.605735 at n = 192 and 288.
- Rung 2a pass condition (4 digits): met (5 digits agree; difference 1.0e-6).
- **Not yet an independent high-precision value.** The 1.0e-6 difference from the published 13-digit value is
  most likely my error: at n = 288 the coefficients for k > 150 sit near 1e-7 without decaying and LM had
  not converged. Second seed and a doubling of the points at fixed n have not been done for this profile.
- The row at n = 128, h = 0.05 is kept as a warning: residual 7e-13 at a lam that is wrong in the 3rd digit.
  Rows must outnumber unknowns by a wide margin in the interior of s; I now use at least 6 rows per unknown.
- The profile needs a far larger basis than the stable one (coefficients fall about one decade per 25-30
  modes), so the discretisation, not the method, limits accuracy. Cost is about 1.5 s per iteration at n = 288
  on the GPU, where the GPU now helps.

Attempt A in the previous update failed because of normalisation and starting shape, not structure. Attempts
C and D (mu = 4) are wrong hypotheses.

### Next
1. A discretisation that resolves the peak (peak-adapted map or two-domain basis), to take lam_1 past 10
   digits and compare with 0.6057337012032.
2. Second unstable profile: the two published values differ at the 7th digit (0.4713248638620 vs 0.47132422).
3. Search for a third.


## Builder update 2026-11-14 — lam_1 to 13 digits with a peak-adapted map; lam_2 in progress

### The fix: Tee-Trefethen sinh map in the Chebyshev variable (`ccf_fast.MAP = (x0, eps)`)
Basis T_k(t(x)), x = 2s-1, t(x) = 2 (asinh((x-x0)/eps) + a2)/(a1+a2) - 1, a1 = asinh((1-x0)/eps),
a2 = asinh((1+x0)/eps). t is analytic on [-1,1], so the origin and far-field structure of the ansatz are
unchanged; resolution is concentrated at x0, where the profile has complex singularities at distance
about eps. Solutions are carried between maps by Chebyshev interpolation of M(s) (`runs/ccf_remap.py`).

### First unstable CCF profile — 13 digits
Normalisation Omega'(0) = -0.1. Peak |Omega| = 1.58997 at y = 3.7542. U'(0) = 1.302866850602.

| n | h (rows) | map (x0, eps) | lam_1 | sup residual | last coefficients |
|---|---|---|---|---|---|
| 96 | 0.025 (880) | (-0.14, 0.1) | 0.60573370120172 | 1.8e-11 | 3.7e-14 |
| 128 | 0.025 (880) | (-0.14, 0.1) | 0.60573370253850 | 2.3e-12 | 2.7e-13 (too few rows for n: off by 1.3e-9) |
| 128 | 0.0125 (1760) | (-0.14, 0.1) | 0.60573370120409 | 2.9e-12 | 2.0e-15 |
| 160 | 0.0125 | (-0.14, 0.1) | 0.60573370120323 | 6.9e-13 | 2.8e-16 |
| 128 | 0.0125 | (-0.10, 0.08) | 0.60573370120275 | 4.2e-12 | 3.1e-15 |
| 128 | 0.0125 | (-0.14, 0.05) | 0.60573370120274 | 5.6e-12 | 7.1e-15 |
| 128 | 0.0125 | (-0.18, 0.13) | 0.60573370120530 | 6.5e-11 | 1.0e-14 |

lam_1 = 0.605733701203 +/- 2e-12 (spread of the well-resolved runs); published 0.6057337012032.
Each solve takes 3-6 s. The plain-basis values in the previous update (0.6057347) were under-resolved,
as suspected; they are superseded.

### Second unstable profile — in progress, not converged
Reached from the first profile by fixing lam near the published value, then freeing it. Fig. 1 and Fig. 3 of
arXiv:2511.22819 (viewed in the browser) show the second profile has the same shape with a peak of about 8.5,
and that for each lam below lam_1 there is a solution non-smooth at the origin whose peak grows as lam falls.
Current state (n = 192, h = 0.0125, map (-0.097, 0.03)): with lam fixed at 0.4713248 the residual fell from
3.4 to 3.0e-5 in 500 iterations; with lam then freed it moved to 0.47132423 (residual 2.3e-5, stopped at the
150-iteration cap, coefficients flat near 3e-5, peak 5.7: under-resolved). This is NOT a result. It is noted
only because the free solve moved away from 0.4713248638620 (2509.14185) toward 0.47132422 (2511.22819).
A run with a sharper map and four times the rows is in progress.


## Builder update 2026-11-14 (second) — lam_2: two more failed routes; unstable branch in a started

### lam_2 attempts (all negative so far)
1. The candidate reported above (lam about 0.471324, residual 2e-5) was another under-sampling artifact: on
   four times the rows its residual was 2.6e2, and 400 LM iterations (n = 256, h = 0.00625, map
   (-0.096, 0.015)) only reached 6e-4 with lam = 0.471339 and a different shape. Withdrawn, including the
   remark that the free solve "moved toward 0.47132422".
2. Family tracking in lam (`runs/ccf_family_track.py`): fixed lam, no origin relation, collocation rows with
   y < 0.5 dropped, map following the peak. It converges at every lam (residual 1e-12 to 1e-8) and
   g(lam) = lam + 2 - 2U'(0) returns to zero near lam = 0.524 with peak 2.5, whereas the published second
   profile is at 0.4713 with peak about 8.5 and the published family (Fig. 3 of 2511.22819) has peak 3.7 at
   lam = 0.51 against my 2.58. So dropping the origin rows changes the object: my tracked family is not the
   true family and its zero of g means nothing. Discarded. (`runs/ccf_family_track_t1.pt`, `_t2.out`.)
   Why, in hindsight (derived here, unchecked): the true family has Theta ~ |y|^mu E(y) with
   E = 1 + e1 |y|^(mu-1) + e2 y^2 + ..., e1 proportional to the non-analytic part of H(|y|^mu), which
   vanishes only at mu = 2. So away from an admissible lam the profile has a relative correction of order
   |y|^(mu-1), first order in y, which an analytic basis cannot follow even approximately.
3. What a correct family solver needs: Theta = r^beta (1-s)^(mu/2) sum_j (1-s)^(j (mu-1)/2) N_j(s), with mu
   unknown and the relation mu = lam/(1 + lam - U'(0)) as a row. Not built.

### First unstable branch in the gCLM parameter a
Started (`runs/gclm_u1_branch.py`): smooth profiles, lam free, origin row, map following the peak.
First look (n = 128, h = 0.025): a = -0.96 gives lam_1 = 0.56416 (residual 1e-8), a = -0.92 gives 0.5229
(residual 3e-6). The branch moves smoothly with a; a higher-resolution run is in progress. I know of no
published values for this branch away from a = -1 (not checked in the literature).


## Builder update 2026-11-14 (third) — exact check of the general-a solver; first unstable branch lam_1(a)

### Exact solution at a = 1/2 (hand-derived, now a test)
G = U' - i Omega is the boundary value of a function analytic in the upper half plane. Taking
G = -(8/3)/(y+i)^2 gives Omega = -(16/3) y/(1+y^2)^2, U = (8/3) y/(1+y^2), and substituting into
Omega + ((1+lam) y + a U) Omega' - Omega U' = 0 with a = 1/2, lam = -2/3 gives identically zero
(the coefficient of each power of y cancels when kappa = 8/3). It also satisfies lam = (1-a)U'(0) - 2.
In the basis it is M = 8/3 (alpha = 3, beta = -1). Discrete residual at the exact solution: 1.5e-13; U, U'
errors 2e-14. Started from a 2e-3 perturbation with lam unknown, LM returns lam = -0.666666666666663.
(From a 2e-2 perturbation it fails: the basin is small. For lam < 0 the basis is also poor away from this
point, because the tail then has a relative y^-2 term, s^(2/alpha), at low order: continuation to a = 0.475
stalls at residual 7e-4. So the solver is validated for general a, but usable only for lam > 0, i.e. a < 0.)
This matches the known value alpha = 1/3 at a = 1/2 in Lushnikov-Silantyev-Siegel (LSS, arXiv:2010.01201).

### Stable branch vs LSS Table 1 (read in the browser from ar5iv)
LSS columns: alpha_e (eigenvalue problem; "3-4 digits for a < 0.3"), alpha and alpha_2 (time-dependent fits,
"3-4 digits"). 1 + lam from my table:

| a | mine | LSS alpha_e | LSS alpha | LSS alpha_2 |
|---|---|---|---|---|
| -1 | 2.1807776629 | - | 2.208 | 2.206 |
| -0.5 | 1.5977112640 | - | 1.603747 | 1.600222 |
| -0.25 | 1.3019558323 | 1.296593455 | 1.303708 | 1.302424 |
| -0.2 | 1.2422022487 | 1.239824952 | 1.243558 | 1.242436 |
| -0.1 | 1.1218375393 | 1.121312899 | 1.122630 | 1.122093 |
| -0.05 | 1.0611326352 | 1.061051829 | 1.061617 | 1.061334 |

Mine lies between alpha_e and alpha_2 at every a where both exist, and within LSS's stated accuracy. No
conflict. (My digits near a = 0 are fewer than shown: see the earlier table's |A - B| column.)

### First unstable branch lam_1(a) — new as far as the literature check found
Smooth profiles, lam unknown, origin row, normalisation Omega'(0) = -0.1, sinh map following the peak,
secant predictor in a (`runs/gclm_u1_branch.py`). "A": n = 192, h = 0.0125 (1760 rows). "B": n = 256,
h = 0.00625 (3520 rows), re-solved from A (`runs/gclm_u1_verify.py`). Residuals of B are 2e-13 to 3e-11.

| a | lam_1 (A) | \|A - B\| | peak \|Omega\| |
|---|---|---|---|
| -1.36 | 0.9803129230147 | 5.7e-13 | 1.1245 |
| -1.30 | 0.9178357743941 | 1.0e-13 | 1.1822 |
| -1.20 | 0.8137482278605 | 2.1e-14 | 1.2928 |
| -1.10 | 0.7097135731416 | 2.7e-14 | 1.4261 |
| -1.00 | 0.6057337012032 | 2.3e-14 | 1.5900 |
| -0.90 | 0.5018107274766 | 4.4e-13 | 1.7964 |
| -0.80 | 0.3979474420328 | 4.8e-11 | 2.0644 |
| -0.74 | 0.3356598768471 | 4.1e-9 | 2.2671 |
| -0.70 | 0.2941484615714 | 6.4e-8 | 2.4260 |

Steps of 0.02 in a are in `runs/gclm_u1_branch.jsonl` and the `.out` files.
- External check: only a = -1 (13 digits against arXiv:2509.14185). Everything else rests on the two
  resolutions and on the solver's exact check at a = 1/2.
- The branch is close to linear: slope d lam_1/da about -1.040, which extrapolates to lam_1 = 0 near
  a = -0.42. The peak grows as a rises. Not computed beyond a = -0.70: accuracy drops (8 digits at -0.74,
  7 at -0.70) and LM slows; at a = -0.64 the run did not converge (residual 3e-6).
- Downward the run stops at a = -1.38, where lam_1 passes through 1. That is a degeneracy of the basis
  (at lam = 1, M = s is a constant Theta, a null direction), not a property of the branch.
- Prior art (Sonnet subagent, `runs/lit_gclm_branches.md`, not verified by me): no unstable branch for
  a != -1 found in arXiv:2201.06780, 2509.14185, 2511.22819, 2610.30492; arXiv:2603.25104 states the
  unstable profiles are known only at a = -1 and gives exact non-smooth profiles with lam = -a for a < 0
  (slope -1, close to the -1.04 found here; worth your attention).

### Questions for the math session
1. Is lam_1(a) expected to end at lam = 0, and at which a? At a = 0 (CLM) the exact family
   g = 2/(1 + C y^(1/c)) has no second smooth member, so the branch must end before a = 0.
2. The exact non-smooth profiles lam = -a of arXiv:2603.25104: are they the limit of this branch in some
   sense (the peak sharpening into their singularity)? lam_1(a) + a is 0.39 at a = -1 and 0.40 at a = -0.7.
3. lam_2: I need the correct family ansatz (see the previous update, item 3) or a better idea for a start.


## Builder update 2026-11-14 (fourth) â€” non-smooth family solver, lam_2 pinned, verification method changed

### What is new in the code
- `gpu_solver/ccf_sing.py`: CCF (a = -1) in Burgers form, -lam Theta + ((1+lam) y - H Theta) Theta' = 0, on the grid
  y = exp(xi), with the origin ansatz
      Theta = r^beta (1-s)^(mu/2) [ N_0(s) + sum_{j=1..J} (1-s)^(j(mu-1)/2) N_j(s) ],
  mu an unknown tied to the solution by the origin relation mu = lam/(1 + lam - U'(0)) (derived here, first
  Taylor order of the Burgers form; NOT checked against the literature â€” the literature agent found no
  published exponent formula). `free=(False, True)`: family at fixed lam, mu unknown. `free=(True, False)`:
  smooth profile, mu = 2, lam unknown.
- Peak-following quadrature grids: `EGrid(cluster=("box", xi_p, c, D))` (flat-top, density D for |xi - xi_p| < c)
  and `ccf.Grid(cluster=(xi_p, w, A))` (Lorentzian). Analytic maps, so the trapezoid rule stays spectral.
- `runs/quad_sharp_pair_check.py`: exact Hilbert pair with a sharp peak (G = i/(z-z0) + i/(z+conj z0)).
  Plain h = 0.0125 gives U' error 5e-2 for a pole at distance 0.05 and 4.6 at distance 0.02; the clustered
  grid gives 1e-11 for both.

### CORRECTION that affects earlier updates: the residual at the collocation rows is not the residual
For the sinh-mapped (peak-concentrated) basis the fit can be 1e-11 at its own rows and far larger between
them. Measured on the third update's branch solutions (fit on plain h = 0.0125), residual on a plain h/2 grid:
a = -1.16..-1.36: 1e-11..1e-12 (fine); a = -1.00: 4e-9; a = -0.80: 3e-8; a = -0.76: 5e-7; a = -0.72: 6e-5;
a = -0.68: 2e-3; a = -0.60: 0.16. The quadrature itself was consistent (U agrees to 1e-12 between h and h/3);
the rows were simply too sparse at the peak for the number of basis functions concentrated there. I also had
a bug in the first clustered sinh grid (dense zone centred off the peak; fixed, `tp` iteration in `ccf.Grid`).
All numbers below are accepted only if the residual on a second, differently placed and denser grid agrees
with the fit residual ("OFF" in the logs, `res_offgrid` in the jsonl).
Consequence for the third update's table: the lam_1(a) VALUES are unchanged to 12 digits by the re-run with
correct clustering (a = -0.80: 0.397947442081, -0.78: 0.377182295841, -0.76: 0.356419756395,
-0.74: 0.335659880995, on-grid and off-grid residual both 2e-11), but the "coarse" column of that table for
a >= -0.74 (0.3356598768, 0.2941484616) was wrong in the 8th-9th digit, as the fine column already showed.

### The non-smooth family between lam_1 and lam_2 (a = -1)
mu(lam) from `runs/sing_track.py` (clustered grid from lam = 0.515 down; n0 = 384, h = 0.00625):
| lam | mu | U'(0) | peak of |Omega| (d = 0.1) | residual (on = off grid) |
|---|---|---|---|---|
| 0.6057337012 (lam_1) | 2 | 1.3028668506 | 1.590 | 5e-13 |
| 0.600 | 1.9912253450 | 1.29867800 | 1.639 | 9e-13 |
| 0.580 | 1.9626565578 | 1.28448217 | 1.841 | 7e-12 |
| 0.560 | 1.9398944358 | 1.27132450 | 2.113 | (plain grid: off-grid 3e-10) |
| 0.530 | 1.9258699104 | 1.25479969 | 2.760 | 1e-10 (minimum of mu is near here) |
| 0.510 | 1.9363088272 | 1.24661226 | 3.516 | 4e-10 |
| 0.500 | 1.9484264786 | 1.24338267 | 4.096 | 1e-9 |
| 0.490 | 1.9646969360 | 1.24059767 | 4.929 | 2e-9 |
| 0.480 | 1.9836894278 | 1.23802664 | 6.242 | 1e-8 |
| 0.47132422777 (lam_2) | 2 | 1.23566211388 | 8.2387 at y = 7.722 | 6e-10 |
So the family is slightly LESS regular than smooth between the two profiles (mu < 2, minimum 1.9259) and
returns to mu = 2 at lam_2. Below lam_2 it goes above 2 (lam = 0.4713: mu = 2.00013; run in progress).
The published picture (Fig. 3 of 2511.22819: peak 3.7 at lam = 0.51) matches my 3.52 at 0.510 / 3.78 at 0.505.

### lam_2 (second unstable CCF profile), lam solved as an unknown
Pinned with mu = 2 fixed and lam free, then re-fitted with the singular blocks REMOVED (smooth basis only):
the smooth-only start already has residual 2e-10, i.e. the profile is smooth.
| n0 | h | grid | lam_2 | residual on-grid | off-grid | last coefficients |
|---|---|---|---|---|---|---|
| 448 | 0.00625 | box c = 0.15, D = 40 | 0.47132422778557 | 1.4e-9 | 1.8e-9 | 6e-12 |
| 640 | 0.00625 | box c = 0.15, D = 40 | 0.47132422776779 | 5.8e-10 | 1.1e-9 | 1e-14 |
| 800 | 0.00625 | box c = 0.15, D = 40 | 0.47132422776818 | 5.8e-10 | 1.1e-9 | 2e-14 |
| 640 | 0.005 | box c = 0.15, D = 40 | 0.47132422776520 | 8.6e-10 | 1.6e-9 | 2e-14 |
| 640 | 0.00625 | box c = 0.30, D = 30 | 0.47132422776261 | 4.6e-10 | 1.2e-9 | 3e-14 |
lam_2 = 0.4713242278 (spread of the four resolved runs 6e-12; I quote 10 digits). Peak |Omega| = 8.2387 at
y = 7.7220, U'(0) = 1.2356621139 (normalisation Omega'(0) = -0.1).
Literature: 2511.22819 gives 0.47132422 (8 digits) â€” agrees. 2509.14185 gives 0.4713248638620 according to my
earlier note, and 0.4703 in the v1 HTML according to the agent â€” mine disagrees with the former in the 7th
digit. The agent also reports that Fig. 5 of 2511.22819 scans lam in [0.471324222, 0.471324226]; if their best
value lies inside that window it differs from mine by 2e-9 to 6e-9. Not resolved; I have not read that figure.
Unexplained: the residual floor (6e-10) is far above the coefficient tail (1e-14).
Withdrawn: the earlier value 0.47132422812 quoted in commit c002021 (fit with singular blocks present and
too few rows on the peak shoulder; off by 3.5e-10).

### First unstable gCLM branch, re-run with correct clustering (n = 256, h = 0.0125, on-grid = off-grid residual)
| a | lam_1 | residual |
|---|---|---|
| -0.80 | 0.397947442081 | 2e-11 |
| -0.70 | (see jsonl, case gclm-u1-branch-upe256) | |
| -0.56 | 0.148956596374 | 1e-10 |
| -0.52 | 0.107507487740 | 9e-11 |
| -0.50 | 0.086790210136 | 9e-11 |
| -0.49 | 0.076433606190 | 1e-10 |
| -0.46 | 0.045372879914 | 3e-10 |
| -0.45 | 0.035022647103 | 4e-10 |
| -0.44 | 0.024674269059 | 8e-10 |
| -0.435 | 0.019500815257 | 7e-10 |
The branch is still almost straight (slope -1.0347 here) and heads for lam_1 = 0 near a = -0.4162. The LM
iteration count grows without bound as lam -> 0 (20 -> 530 per step), so the Jacobian is degenerating there;
I do not know why yet. Withdrawn: every row of the earlier runs upc288/upw288 (clustering off the peak).

### Negative results this round
- Far-field blocks s^(2j(1+lam)) P_j(s) (`ccf_fast.FAR`) do not change anything: the mid-spectrum plateau was
  not a far-field singularity. Left in the code, off by default.
- Stepping the singular ansatz THROUGH mu = 2 is slow (even-j blocks are redundant with N_0 there). Restart
  from the pinned smooth profile on the other side instead (`START=` in `runs/sing_track.py`).

### Questions for the math session
1. Is mu = lam/(1 + lam - U'(0)) and the exponent set mu + j(mu-1) + 2k right? The solver is consistent with
   it (residual 1e-12 near lam_1 with four blocks), which is evidence, not a proof.
2. mu < 2 between lam_1 and lam_2 and mu > 2 just below lam_2. Does the sign pattern alternate, and is there
   a reason the excursion is so small (min mu = 1.926)?
3. Why does the a-branch linearisation degenerate as lam_1 -> 0 (a -> -0.416)?
