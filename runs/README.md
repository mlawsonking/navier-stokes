# runs/

- `ledger.jsonl` â€” one JSON line per solve: case, basis size, seed, initial guess, result, error vs reference,
  residuals, iterations, wall time, pass flag, commit, device, timestamp. Runs made before the first commit show
  `f6897e4+dirty` (code then uncommitted). The code at commit 9b8feeb plus the trial scripts here reproduces them.
- `ledger_attempt1_normal_eqs.jsonl` â€” first pass with LM via normal equations (superseded by the QR solve).
- `representation_check.py` / `.txt` â€” best linear fit of the basis to the exact profiles with lam fixed.
- `smooth_trial.py` â€” derivative-row attempt; stopped before any result (too slow).
- `analytic_trial.py` â€” complex off-axis rows. The first version (Im 0.25 and 0.5) put rows outside the
  analyticity strip and failed everywhere; it is logged as case `1a-burgers-analytic`. The corrected version
  (Im 0.1 and 0.2, real-only solve first) is `1a-burgers-analytic2`.

Reproduce:
    .venv/Scripts/python run_rung1.py
    .venv/Scripts/python run_rung1.py refine
    .venv/Scripts/python runs/analytic_trial.py 48 96
    .venv/Scripts/python runs/representation_check.py
    .venv/Scripts/python tests/test_gpu_solver.py
- `lam_scan.py` / `lam_scan.jsonl` — lam fixed on a grid, coefficients solved; the V-shape diagnostic.
- `origin_check.py` — origin series vs finite fit and hand formulas.
- `origin_trial.py` / `.out` — origin rows with white-noise starts (mostly fails to converge; kept as a record).
- `origin_continuation.py` / `.out` — origin rows, decaying-noise multi-start at n = 48, continuation to 96, 192. PASS.
- `lit_2a_notes.md` — literature notes for rung 2a from a Sonnet subagent; not verified by the builder.
- `lit_prior_art_ccf.md` — prior-art notes (Sonnet subagent); not verified by the builder.
- `ccf_operator_check.py` — CCF quadrature operators vs the exact CLM pair.
- `ccf_first.py`, `ccf_continue.py` — CCF multi-start and continuation (env H, L, LC, NGL set the quadrature).
- `ccf_scan.py`, `ccf_scan2.py`, `ccf_scan.jsonl`, `ccf_scan_*.out` — fixed-lam scans (stable V; unstable valley).
- `*.pt` — saved parameter vectors [lam, c_0..c_(n-1)] (torch.load).
- `ccf_fixed.py`, `ccf_multistart.py`, `ccf_k2.py`, `ccf_k2_scan.py`, `ccf_nu_continuation.py`, `ccf_nu_multistart.py`,
  `ccf_picard.py` — the unstable-profile attempts A-F in HANDOFF (all negative). `ccf_fast_check.py` — fast vs old solver.
- `gclm_branch.py`, `gclm_branch.jsonl`, `gclm_*.out` — stable branch of the gCLM family continued in a.
- `ccf_unstable1.py`, `ccf_u_refine.py`, `ccf_u1_*.pt` — first unstable CCF profile (lam_1 = 0.605735 at n = 192, 288).
  `ccf_scale_test.py` — a representation test that turned out unreliable (ill-conditioned fit); ignore.
