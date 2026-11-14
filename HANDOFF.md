# HANDOFF — gpu-solver branch (builder session on Bobbo, RTX 4080 SUPER)

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
`gpu_solver/` — torch, float64, on CUDA.
- `lm.py`: Levenberg–Marquardt, Jacobians from `torch.func.jacrev`. The damped step is solved by QR of the
  column-scaled augmented Jacobian. (Normal equations were the first version; they square the condition number.
  Those runs are kept in `runs/ledger_attempt1_normal_eqs.jsonl`.)
- `burgers.py` (1a): ansatz `U = -y s N(s)`, `s = (1+y^2)^(-a/2)`, `a = 1/(1+lam)`,
  `N(s) = 1 + s(1-s) * sum_k c_k T_k(2s-1)`. So U is odd and analytic at y = 0.
  `N(1) = 1` fixes U'(0) = -1, and `N(0) = 1` fixes the far field U ~ -y^(lam/(1+lam)), i.e. C = 1 in
  y = -U - C U^(1+1/lam). The far-field factor excludes the trivial solution U = -y. lam is an unknown.
  U'''(0) = 3a(1 - P(1)), so U'''(0) = 0 is equivalent to sum_k c_k = 1 (one extra residual row).
  Optional rows: the residual at complex points y = x + i*{0.1, 0.2}, x in [0.05, 4]. The exact profiles'
  nearest branch points are at |Im y| ≈ 0.385 (lam = 1/2) and ≈ 0.38 (lam = 1/4).
- `clm.py` (1b): y = tan(θ/2), W = sum_{k<=K} b_k sin kθ, y W' = sin θ · F'(θ),
  H W = -sum b_k cos kθ + sum b_k (-1)^k (Cayley map). Normalisation W'(0) = -2. c is fixed at 1 or left free.
- `hilbert.py`: FFT Hilbert transform on R via the Cayley map.
- `tests/test_gpu_solver.py`: 4 checks, all pass (three closed-form Hilbert pairs to < 1e-12; closed form vs FFT;
  the exact profiles have zero residual).

## Results (rung 1)

### 1b CLM — PASS
Reference: W = -2y/(1+y^2), c = 1 (exact, DIRECTION.md).

| run | K modes | seeds | c found | sup residual, \|y\|<=50 | sup profile error, \|y\|<=50 |
|---|---|---|---|---|---|
| c fixed = 1 | 16, 32 | 0, 1 | 1 | <= 8.9e-16 | <= 1.2e-15 |
| c free, c0 = 0.8 and 1.3 | 16, 32 | 0, 1 | 1 - 7.1e-14 (worst) | <= 5.5e-15 | <= 1.6e-13 |

The pass condition (residual < 1e-8, profile error < 1e-6) holds in all 12 runs. With c free, c = 1 is recovered.
Caveat: the exact profile is a single sine mode in this basis, so this tests the nonlocal operator and the
Newton machinery more than the resolution.

### 1a Burgers, lam free (stable branch) — PASS
Reference: lam = 1/2, y = -U - U^3 (exact).

| n basis | colloc | lam0 | lam found | \|lam - 1/2\| | sup profile err, \|y\|<=20 | sup residual |
|---|---|---|---|---|---|---|
| 24 | 48 | 0.45, 0.55, 0.70 | 0.4999358460 | 6.4e-5 | 2.0e-4 | 2.3e-7 (basis too small: fails) |
| 48 | 96 | 0.45, 0.55, 0.70 | 0.4999999974 | 2.6e-9 | 8.2e-9 | 2.8e-12 |
| 96 | 192 | 0.55 | 0.500000000000 | 2.5e-13 | 7.7e-13 | 1.1e-15 |
| 192 | 384 | 0.55 | 0.500000000001 | 1.4e-12 | 4.4e-12 | 1.8e-15 |
| 96, + complex rows | 192 | 0.55 | 0.500000000000 | 1.2e-14 | 4.1e-14 | 1.3e-15 (complex 2.6e-15) |

Two seeds each, identical to the digits shown. Wall time 0.2–2 s per solve. lam is stable under doubling the
basis (48 → 96 → 192).

### 1a Burgers with U'''(0) = 0 (unstable branch, lam = 1/4) — FAIL (threshold 1e-6)
Reference: lam = 1/4, y = -U - U^5 (exact).

| n | rows | lam0 | lam found | \|lam - 1/4\| | profile err vs y = -U - U^5 |
|---|---|---|---|---|---|
| 24 | real | 0.2, 0.3 | 0.2501506 | 1.5e-4 | 4.9e-4 |
| 48 | real | 0.3 | 0.2499963332 | **3.7e-6 (best)** | 1.2e-5 |
| 48 | real + complex | 0.3 | 0.2500043433 | 4.3e-6 | 1.4e-5 |
| 96 | real | 0.3 | 0.2561 / 0.2580 (seed 0/1) | 6–8e-3 | 2–3e-2 |
| 192 | real | 0.3 | 0.2707 / 0.2757 | 2–3e-2 | 7–8e-2 |
| 96 | real + complex | 0.3 | 0.2841 / 0.2697 | 2–3e-2 | 7e-2 – 1e-1 |

The error gets **worse** as the basis is refined. This is not convergence.

Diagnosis, partly measured (`runs/representation_check.txt`): with lam fixed at 1/4, the basis fits the exact
profile to 4.2e-11 (n = 48) and 4.5e-13 (n = 96). Representation is therefore not the problem; the residual
pins lam down only weakly. For lam ≠ 1/(2i) the exact sublinear solutions contain |U|^(1+1/lam), which is
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
- first unstable CCF profile: lam_1 ≈ 0.6057 (originally Wang et al., recovered by Eggers and Fontelos);
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


## Builder update 2026-11-13 � rung 1 now passes in full (run on this machine, RTX 4080 SUPER)

Followed the math session's order of work. Records: `runs/lam_scan.jsonl`, `runs/ledger.jsonl`
(cases `1a-burgers-origin*`), raw output in `runs/origin_continuation.out`.

### 1. lam-scan diagnostic � prediction confirmed
lam fixed on 21 values in [0.24, 0.26], coefficients solved (old rows: collocation + sum c = 1), all 21
in one batched solve (3.5 s per n).

| n | min \|r\|_2 (at lam = 0.250) | \|r\|_2 at 0.249 / 0.251 | \|r\|_2 at 0.240 / 0.260 | slope d\|r\|/dlam |
|---|---|---|---|---|
| 48 | 3.0e-9 | 7.0e-8 / 7.8e-8 | 4.4e-7 / 1.2e-6 | about 7e-5 |
| 96 | 1.1e-10 | 4.8e-9 / 5.4e-9 | 2.7e-8 / 8.9e-8 | about 5e-6 |

A V centred at 0.25 whose slope drops about 14x from n = 48 to n = 96. (The V is not symmetric.)

### 2. Origin Taylor rows � 1a unstable PASS, 1a stable still PASS
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

### 4. Literature for 2a (Sonnet subagent, NOT run or checked by me) � `runs/lit_2a_notes.md`
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
