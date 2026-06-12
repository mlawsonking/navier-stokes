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

## Added experiment: arbitrary first-handoff search

Run:

```bash
PYTHONPATH=. python experiments/optimize_first_handoff.py --samples 40000 --eta 0.8 --min-f1-energy 0.2 --min-d-ratio 0.3 --max-l-over-d 5
PYTHONPATH=. python experiments/optimize_first_handoff.py --samples 20000 --complex --eta 0.8 --min-f1-energy 0.2 --min-d-ratio 0.3 --max-l-over-d 5
```

Purpose: vary arbitrary first-shell amplitudes

```text
e1 -> (0,A,B), e2 -> (C,0,D), e3 -> (E,F,0)
```

and test whether strong balanced first-handoffs can drive the second-stage radius/polarization/vorticity defects near zero.

Current constrained random-search result: under balanced, non-tiny forward-transfer constraints, the best real sample found had vorticity line miss ≈ 0.032 and radius-relative variance ≈ 0.063. The best complex sample found had vorticity line miss ≈ 0.035 and radius-relative variance ≈ 0.062. This is evidence, not proof, for a positive first-handoff defect under nondegenerate transfer constraints.

## Robust metric correction and helical scan (latest)

Earlier optimization used global collision miss over every formal multi-path output.  That metric was exploitable: tiny off-cycle perturbations created many fake collision groups where one path dominated and the other was negligible, diluting the average defect.

New robust script:

```bash
PYTHONPATH=. python experiments/robust_optimize_first_handoff.py --runs 3 --eta 0.8 --min-second-ratio 0.1 --min-f1-energy 0.2 --min-d-ratio 0.3 --max-l-over-d 5 --json robust_de_real_output.json
```

It counts only comparable path collisions where the second-largest path energy is at least a chosen fraction of the largest.  Under this metric, the best real search found robust vorticity line miss ≈ 0.049 and radius-relative variance ≈ 0.060.  A short complex search found robust vorticity line miss ≈ 0.051 and radius-relative variance ≈ 0.061.

New classifier:

```bash
PYTHONPATH=. python experiments/classify_minima.py robust_de_real_output.json robust_de_complex_output.json --cycle-scan --eta 0.8 --min-f1-energy 0.2 --min-d-ratio 0.3 --max-l-over-d 5
```

The robust optimizer no longer collapses exactly to the pure 3-cycle; it keeps roughly 92–93% of parameter energy in one 3-cycle and 7–8% off-cycle.  This suggests the best escape tries to perturb a cyclic parent-polarization pattern while still producing comparable collision defects.

New helical scan:

```bash
PYTHONPATH=. python experiments/helical_line_bundle_scan.py --min-second-ratio 0.1 --json helical_line_scan_output.json
```

This enumerates all 4096 pure-helicity sign assignments on F1 ∪ D ∪ L.  Best all-same-helicity assignments have robust vorticity miss ≈ 0.00483 and radius-relative variance ≈ 0.0385, with output helicity purity ≈ 0.995.  This is an important escape candidate: helical line bundles reduce polarization defect strongly, but they do not eliminate radius fan / heat-detuning defect.  The theorem target therefore shifts from “polarization defect alone” to “helicity can reduce polarization defect, but full-NS leakage still forces radius spread or other maintenance cost.”

## Latest experiment: helical radius closure

The helical line scan found an important escape: pure same-helicity assignments nearly eliminate the first-fan polarization/vorticity line-bundle defect.  The new experiment `experiments/helical_radius_closure.py` tests what happens if that escape is accepted and the output fan is iterated.

Two modes were tested:

1. `--project-helical`: after every closure stage, each output mode is projected back to its dominant helical component.  This is deliberately favorable to the helical scalar-cascade escape.
2. full-vector mode: all vector output components are retained.

Main observation: the helical-projected escape keeps polarization defect tiny and helicity purity essentially 1, but the active modes collapse strongly toward a single direction/ray while normalized interaction coefficients fall sharply.  This supports a revised dichotomy:

- non-helical / real scalar profiles pay polarization and vorticity line-bundle defect;
- pure-helical scalar profiles avoid that defect but appear to drift toward near-ray geometry with coefficient depletion.

Representative helical-projected run:

| stage | modes | radius classes | radius range | rel. radius var | direction linearity | coeff mean |
|---|---:|---:|---:|---:|---:|---:|
| stage1 | 31 | 8 | 9–21 | 0.04765 | 0.91378 | 9.34e-2 |
| stage2 | 127 | 26 | 34–74 | 0.01459 | 0.95571 | 1.42e-3 |
| stage3 | 250 | 53 | 134–230 | 0.00757 | 0.97977 | 2.31e-4 |
| stage4 | 250 | 68 | 588–794 | 0.00343 | 0.98658 | 5.69e-5 |
| stage5 | 250 | 84 | 2523–2946 | 0.00070 | 0.99681 | 1.37e-5 |

The coefficient statistic is a normalized projected-NS output size: roughly `|B(p,q)| / (N |a_p||a_q|)`.  Its rapid decay is consistent with the near-parallel / bounded-offset degeneracy mechanism, although this is still an experimental diagnostic rather than a theorem.

Updated theorem target:

> Full-NS first-handoff leakage forces either (a) non-helical polarization/vorticity line-bundle defect, or (b) pure-helical coherence that evolves toward near-ray geometry and loses nondegenerate `O(N)` triad strength, or (c) broad non-helical fan growth with heat/radius maintenance burden.

