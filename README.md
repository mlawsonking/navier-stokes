# Navier--Stokes Cascade Research Sandbox

This is a finite-dimensional experimental sandbox for the Navier--Stokes cascade/leakage program developed in the chat.

It does **not** prove Navier--Stokes regularity or blow-up. It computes exact Fourier-symbol facts for finite mode fans:

- Leray projection and the exact projected bilinear symbol
- permutation-shell seed amplitudes
- forced first leakage fan
- radius fan growth
- path-collision reports
- velocity/vorticity line-bundle defect
- helical-coordinate diagnostics


## Fresh-chat handoff

Read `HANDOFF.md` first. For a new ChatGPT project/chat, upload the zip and paste `NEXT_CHAT_PROMPT.md`.

## Run

```bash
python experiments/first_fan.py
```

## Main files

- `navier_stokes_cascade/ns_symbol.py` — projection, bilinear map, vorticity polarization, angle/line-defect metrics.
- `navier_stokes_cascade/mode_sets.py` — permutation shells and symmetric seed/fan amplitudes.
- `navier_stokes_cascade/defects.py` — interaction enumeration, collision reports, radius variance.
- `navier_stokes_cascade/helical.py` — helical basis and helicity diagnostics.
- `experiments/first_fan.py` — first experiment and sanity checks.

## Current theorem target

Universal First-Handoff Defect Theorem:

Given arbitrary strong 3D first-shell transfer, the full Navier--Stokes leakage layer should force either radius variance, polarization/vorticity covariance defect, suppression loss, or geometric degeneration.

This repo currently verifies the symmetric seed case and surfaces the next computational questions.

## Latest continuation notes

The experiments have now separated the first-handoff obstruction into regimes:

1. **Real / non-helical cyclic geometries**: robust comparable-collision vorticity miss stays near `0.05`, with radius relative variance near `0.06`.
2. **Pure-helical line-bundle geometries**: polarization/vorticity miss can collapse near zero, but the closure drifts toward a near-ray frequency set.
3. **Full-vector helical closure**: if the output is not projected back to a pure helical line, helicity purity decays and robust vorticity miss grows rapidly.

The new helical closure diagnostics support the following working trichotomy:

```text
real/scalar cascade          -> robust line-bundle/vorticity defect
full-vector helical fan      -> growing polarization/vorticity defect
projected pure-helical fan   -> near-ray collapse and normalized coefficient depletion
```

Additional experiment files:

- `experiments/robust_optimize_first_handoff.py` — robust optimization using only comparable multi-path collisions.
- `experiments/classify_minima.py` — classifies low-defect optimizer minima against cyclic/off-cycle parent-polarization patterns.
- `experiments/helical_line_bundle_scan.py` — scans pure-helicity sign assignments on the first fan.
- `experiments/helical_radius_closure.py` — iterates projected/full-vector helical closures and reports radius/linearity/coefficient diagnostics.
- `experiments/helical_closure_sweep.py` — checks whether projected-helical near-ray collapse is stable across cutoffs and mode caps.
- `experiments/near_ray_coefficient_scan.py` — direct controlled scan showing normalized NS coefficients decay like `N^-1` for bounded-defect near-ray mode pairs, while transverse control interactions remain `O(1)`.

Key recent outputs:

- `helical_closure_sweep.txt`
- `near_ray_coefficient_scan.txt`
- `helical_projected_nocap_rel1e4.txt`

The strongest current theorem target is now the **Helical Escape Trichotomy**:

> A strong leakage-absorbing first handoff either develops robust polarization/vorticity defect, or stays nearly pure-helical and collapses toward near-ray geometry, causing normalized triad coefficient depletion, or suppresses a significant fraction of forced output.

This is still far from a Clay proof. The immediate finite theorem target is to prove the controlled near-ray coefficient depletion formula and then connect it to the observed projected-helical closure collapse.
