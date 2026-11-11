"""Controls for the near-ray collapse seen in experiments/helical_radius_closure.py.

One-sided (positive-octant) mode sets collapse toward a ray under repeated
sums even with no Navier--Stokes structure at all.  Imposing the reality
condition u(-k) = conj(u(k)) removes the collapse.
"""
from collections import defaultdict

import numpy as np

from experiments.helical_radius_closure import (
    aggregate_outputs,
    all_pair_interactions,
    pure_helical,
    stage_report,
)
from navier_stokes_cascade.mode_sets import add_modes, perm_shell, with_conjugates
from navier_stokes_cascade.ns_symbol import ns_bilinear

SEEDS = [k for t in ((1, 1, 0), (2, 1, 1), (2, 1, 0)) for k in perm_shell(t)]
DEPTH = 3


def _closure_linearity(active, depth=DEPTH):
    for _ in range(depth):
        inters = all_pair_interactions(active)
        active, _meta = aggregate_outputs(
            inters, project_helical=True, rel_cut=1e-6, max_modes=250, normalize=True
        )
    return stage_report("x", active)["direction_linearity"]


def _scalar_sum_linearity(depth=DEPTH):
    """No physics: w(p+q) += w(p) w(q), keep the 250 heaviest modes."""
    w = {k: 1.0 for k in SEEDS}
    for _ in range(depth):
        new = defaultdict(float)
        for p, a in w.items():
            for q, b in w.items():
                new[add_modes(p, q)] += a * b
        w = dict(sorted(new.items(), key=lambda t: -t[1])[:250])
    cov = np.zeros((3, 3))
    for k, v in w.items():
        kh = np.asarray(k, float) / np.linalg.norm(k)
        cov += v * np.outer(kh, kh)
    return float(np.linalg.eigvalsh(cov / sum(w.values()))[-1])


def test_symbol_respects_conjugation():
    p, q = (2, 1, 0), (1, 0, 2)
    a, b = pure_helical(p, -1), pure_helical(q, +1)
    out = ns_bilinear(p, a, q, b)
    neg = ns_bilinear((-2, -1, 0), np.conjugate(a), (-1, 0, -2), np.conjugate(b))
    # ns_bilinear drops the global factor -i, so its output is anti-conjugate
    # symmetric.  Restoring -i gives the usual u(-k) = conj(u(k)); the dropped
    # factor is a uniform phase per closure stage and no diagnostic depends on it.
    assert np.allclose(neg, -np.conjugate(out))
    assert np.allclose(-1j * neg, np.conjugate(-1j * out))


def test_one_sided_collapse_needs_no_navier_stokes():
    one_sided = _closure_linearity({k: pure_helical(k, -1) for k in SEEDS})
    assert one_sided > 0.95
    assert _scalar_sum_linearity() > 0.95


def test_real_field_does_not_collapse_to_a_ray():
    real = _closure_linearity(with_conjugates({k: pure_helical(k, -1) for k in SEEDS}))
    assert real < 0.8
