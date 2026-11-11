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
(empty)
