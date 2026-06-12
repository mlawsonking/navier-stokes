#!/usr/bin/env python3
"""Random/heuristic search for universal first-handoff defect.

This experiment varies arbitrary first-shell parent amplitudes
    e1 -> (0,A,B), e2 -> (C,0,D), e3 -> (E,F,0)
(real or complex), generates the induced first clean shell F1, first leakage shell L,
and desired shell D, then measures second-stage collision line-bundle defects.

It is not a proof. It searches for escape geometries where strong 3D transfer has
small radius/polarization/vorticity defect.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from navier_stokes_cascade.defects import interactions_between, radius_energy, radius_variance, group_by_output
from navier_stokes_cascade.mode_sets import add_modes
from navier_stokes_cascade.ns_symbol import Array, Mode, ns_bilinear, norm2, norm, one_line_miss_fraction, vorticity_polarization

E_MODES: list[Mode] = [(1,0,0), (0,1,0), (0,0,1)]


def normalize_parent(F0: Dict[Mode, Array]) -> Dict[Mode, Array]:
    total = sum(norm2(v) for v in F0.values())
    if total <= 1e-30:
        return F0
    scale = 1.0 / math.sqrt(total)
    return {k: scale * v for k, v in F0.items()}


def parent_from_params(x: np.ndarray, *, complex_vars: bool = False) -> Dict[Mode, Array]:
    if complex_vars:
        z = x[:6] + 1j * x[6:12]
    else:
        z = x[:6].astype(np.complex128)
    A,B,C,D,E,F = z
    F0 = {
        (1,0,0): np.array([0,A,B], dtype=np.complex128),
        (0,1,0): np.array([C,0,D], dtype=np.complex128),
        (0,0,1): np.array([E,F,0], dtype=np.complex128),
    }
    return normalize_parent(F0)


def aggregate_interactions(pairs: Iterable[tuple[Mode, Array, Mode, Array]], keep_sorted: list[int]) -> Dict[Mode, Array]:
    out: Dict[Mode, Array] = {}
    for p,a,q,b in pairs:
        m = add_modes(p,q)
        if sorted(m) == keep_sorted:
            v = ns_bilinear(p,a,q,b)
            if norm(v) > 1e-12:
                out[m] = out.get(m, np.zeros(3, dtype=np.complex128)) + v
    return out


def induced_fan(F0: Dict[Mode, Array]) -> dict[str, Dict[Mode, Array]]:
    items0 = list(F0.items())
    # F1: unordered distinct parent pairs e_i + e_j = P(1,1,0)
    f1_pairs = []
    for i in range(len(items0)):
        for j in range(i+1, len(items0)):
            p,a = items0[i]
            q,b = items0[j]
            f1_pairs.append((p,a,q,b))
    F1 = aggregate_interactions(f1_pairs, [0,1,1])

    # L: cross F0 x F1 landing in P(2,1,0)
    l_pairs = [(p,a,q,b) for p,a in F0.items() for q,b in F1.items()]
    L = aggregate_interactions(l_pairs, [0,1,2])

    # D: unordered distinct F1 pairs landing in P(2,1,1)
    items1 = list(F1.items())
    d_pairs = []
    for i in range(len(items1)):
        for j in range(i+1, len(items1)):
            p,a = items1[i]
            q,b = items1[j]
            d_pairs.append((p,a,q,b))
    D = aggregate_interactions(d_pairs, [1,1,2])
    return {"F0": F0, "F1": F1, "L": L, "D": D}


def global_collision_miss(interactions, *, vorticity: bool = False) -> tuple[float, float, int, int]:
    """Return (miss_fraction over collision outputs, collision_energy, num_collision_modes, num_paths)."""
    total_energy = 0.0
    missed = 0.0
    num_modes = 0
    num_paths = 0
    for m, group in group_by_output(interactions).items():
        if len(group) < 2:
            continue
        vecs = [g.output for g in group]
        if vorticity:
            vecs = [vorticity_polarization(m, v) for v in vecs]
        e = sum(norm2(v) for v in vecs)
        if e <= 1e-14:
            continue
        miss = one_line_miss_fraction(vecs)
        total_energy += e
        missed += miss * e
        num_modes += 1
        num_paths += len(group)
    return (missed / total_energy if total_energy > 1e-14 else 0.0, total_energy, num_modes, num_paths)


def second_stage_interactions(fam: dict[str, Dict[Mode, Array]]):
    F1,D,L = fam["F1"], fam["D"], fam["L"]
    blocks = []
    if L and F1: blocks += interactions_between(L,F1)
    if D and F1: blocks += interactions_between(D,F1)
    if L: blocks += interactions_between(L,L)
    if D: blocks += interactions_between(D,D)
    if D and L: blocks += interactions_between(D,L)
    return blocks


@dataclass
class SampleMetrics:
    idx: int
    objective: float
    vel_miss: float
    vort_miss: float
    collision_energy: float
    collision_modes: int
    collision_paths: int
    radius_rel_var: float
    radius_classes: int
    f1_energy: float
    f1_minmax: float
    d_energy: float
    l_energy: float
    l_over_d: float
    params: list[float]


def evaluate(x: np.ndarray, *, idx: int = 0, complex_vars: bool = False, eta_balance: float = 0.20, min_f1_energy: float = 1e-3, min_d_ratio: float = 0.0, max_l_over_d: float = 1e9) -> SampleMetrics | None:
    F0 = parent_from_params(x, complex_vars=complex_vars)
    fam = induced_fan(F0)
    F1,D,L = fam["F1"], fam["D"], fam["L"]
    if len(F1) < 3:
        return None
    f1_norms = np.array([norm(v) for v in F1.values()], dtype=float)
    f1_energy = float(np.sum(f1_norms**2))
    if f1_energy < min_f1_energy:
        return None
    f1_minmax = float(np.min(f1_norms) / (np.max(f1_norms) + 1e-15))
    if f1_minmax < eta_balance:
        return None
    inters = second_stage_interactions(fam)
    if not inters:
        return None
    vel_miss, coll_e, coll_modes, coll_paths = global_collision_miss(inters, vorticity=False)
    vort_miss, _, _, _ = global_collision_miss(inters, vorticity=True)
    e_rad = radius_energy(inters)
    mean_r = sum(r*e for r,e in e_rad.items()) / (sum(e_rad.values()) + 1e-15)
    rad_rel = radius_variance(e_rad) / ((mean_r**2) + 1e-15)
    d_energy = sum(norm2(v) for v in D.values())
    l_energy = sum(norm2(v) for v in L.values())
    l_over_d = float(l_energy / (d_energy + 1e-15))
    d_ratio = float(d_energy / ((f1_energy ** 2) + 1e-15))
    if d_ratio < min_d_ratio:
        return None
    if l_over_d > max_l_over_d:
        return None
    # Objective: escape score. We want to find low vorticity defect + low radius spread + not-too-small leakage.
    # Small objective means a possible escape from our proposed obstruction.
    objective = float(vort_miss + 0.15*rad_rel)
    return SampleMetrics(
        idx=idx,
        objective=objective,
        vel_miss=float(vel_miss),
        vort_miss=float(vort_miss),
        collision_energy=float(coll_e),
        collision_modes=int(coll_modes),
        collision_paths=int(coll_paths),
        radius_rel_var=float(rad_rel),
        radius_classes=len(e_rad),
        f1_energy=float(f1_energy),
        f1_minmax=float(f1_minmax),
        d_energy=float(d_energy),
        l_energy=float(l_energy),
        l_over_d=float(l_over_d),
        params=[float(z) for z in np.real(x).tolist()],
    )


def random_search(n: int, *, seed: int, complex_vars: bool, eta_balance: float, min_f1_energy: float, min_d_ratio: float, max_l_over_d: float) -> list[SampleMetrics]:
    rng = np.random.default_rng(seed)
    best: list[SampleMetrics] = []
    dim = 12 if complex_vars else 6
    accepted = 0
    for i in range(n):
        x = rng.normal(size=dim)
        m = evaluate(x, idx=i, complex_vars=complex_vars, eta_balance=eta_balance, min_f1_energy=min_f1_energy, min_d_ratio=min_d_ratio, max_l_over_d=max_l_over_d)
        if m is None:
            continue
        accepted += 1
        best.append(m)
        best.sort(key=lambda z: z.objective)
        if len(best) > 20:
            best = best[:20]
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--complex", action="store_true", dest="complex_vars")
    ap.add_argument("--eta", type=float, default=0.25, help="min child norm / max child norm threshold")
    ap.add_argument("--min-f1-energy", type=float, default=1e-3)
    ap.add_argument("--min-d-ratio", type=float, default=0.0, help="require D_energy / F1_energy^2 >= this")
    ap.add_argument("--max-l-over-d", type=float, default=1e9)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    best = random_search(args.samples, seed=args.seed, complex_vars=args.complex_vars, eta_balance=args.eta, min_f1_energy=args.min_f1_energy, min_d_ratio=args.min_d_ratio, max_l_over_d=args.max_l_over_d)
    print("Random search for low-defect strong first handoffs")
    print(f"samples={args.samples} complex={args.complex_vars} eta={args.eta} min_f1={args.min_f1_energy} min_d_ratio={args.min_d_ratio} max_l_over_d={args.max_l_over_d} best_count={len(best)}")
    if not best:
        print("No accepted samples.")
        return
    for rank, m in enumerate(best[:10], 1):
        print("-"*88)
        print(f"rank={rank} idx={m.idx} objective={m.objective:.6g}")
        print(f"  vel_miss={m.vel_miss:.6g} vort_miss={m.vort_miss:.6g} collision_energy={m.collision_energy:.6g}")
        print(f"  radius_rel_var={m.radius_rel_var:.6g} radius_classes={m.radius_classes}")
        print(f"  f1_energy={m.f1_energy:.6g} f1_minmax={m.f1_minmax:.6g}")
        print(f"  d_energy={m.d_energy:.6g} l_energy={m.l_energy:.6g} l_over_d={m.l_over_d:.6g}")
        print(f"  params={np.array2string(np.array(m.params), precision=5, suppress_small=True)}")
    if args.json:
        args.json.write_text(json.dumps([asdict(m) for m in best], indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
