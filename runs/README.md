# runs/

- `ledger.jsonl` — one JSON line per solve: case, basis size, seed, initial guess, result, error vs reference,
  residuals, iterations, wall time, pass flag, commit, device, timestamp. Runs made before the first commit show
  `f6897e4+dirty` (code then uncommitted). The code at commit 9b8feeb plus the trial scripts here reproduces them.
- `ledger_attempt1_normal_eqs.jsonl` — first pass with LM via normal equations (superseded by the QR solve).
- `representation_check.py` / `.txt` — best linear fit of the basis to the exact profiles with lam fixed.
- `smooth_trial.py` — derivative-row attempt; stopped before any result (too slow).
- `analytic_trial.py` — complex off-axis rows. The first version (Im 0.25 and 0.5) put rows outside the
  analyticity strip and failed everywhere; it is logged as case `1a-burgers-analytic`. The corrected version
  (Im 0.1 and 0.2, real-only solve first) is `1a-burgers-analytic2`.

Reproduce:
    .venv/Scripts/python run_rung1.py
    .venv/Scripts/python run_rung1.py refine
    .venv/Scripts/python runs/analytic_trial.py 48 96
    .venv/Scripts/python runs/representation_check.py
    .venv/Scripts/python tests/test_gpu_solver.py
