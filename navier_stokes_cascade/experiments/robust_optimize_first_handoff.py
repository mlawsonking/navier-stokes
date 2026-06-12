#!/usr/bin/env python3
"""Robust adversarial search for first-handoff defects.

This fixes an important metric artifact in earlier searches: tiny off-cycle components
can create many formal path collisions where one path dominates and the other is
negligible. Such nearly one-path collisions should not dilute the line-bundle defect.

We therefore count only *comparable* collision groups whose second-largest path energy
is at least a fixed fraction of the largest path energy.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution, minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.optimize_first_handoff import parent_from_params, induced_fan, second_stage_interactions
from navier_stokes_cascade.defects import group_by_output, radius_energy, radius_variance
from navier_stokes_cascade.ns_symbol import norm2, one_line_miss_fraction, vorticity_polarization, norm


@dataclass
class RobustMetrics:
    objective: float
    robust_vel_miss: float
    robust_vort_miss: float
    comparable_collision_energy: float
    comparable_collision_modes: int
    comparable_collision_paths: int
    all_collision_modes: int
    all_collision_paths: int
    radius_rel_var: float
    radius_classes: int
    f1_energy: float
    f1_minmax: float
    d_energy: float
    l_energy: float
    l_over_d: float
    d_ratio: float
    params: list[float]


def comparable_collision_miss(interactions, *, vorticity: bool, min_second_ratio: float) -> tuple[float, float, int, int, int, int]:
    total_energy = 0.0
    missed = 0.0
    comp_modes = 0
    comp_paths = 0
    all_modes = 0
    all_paths = 0
    for m, group in group_by_output(interactions).items():
        if len(group) < 2:
            continue
        all_modes += 1
        all_paths += len(group)
        energies = np.array([norm2(g.output) for g in group], dtype=float)
        if len(energies) < 2 or np.max(energies) <= 1e-18:
            continue
        e_sorted = np.sort(energies)[::-1]
        if e_sorted[1] / (e_sorted[0] + 1e-30) < min_second_ratio:
            continue
        vecs = [g.output for g in group]
        if vorticity:
            vecs = [vorticity_polarization(m, v) for v in vecs]
        e = float(np.sum([norm2(v) for v in vecs]))
        if e <= 1e-14:
            continue
        miss = one_line_miss_fraction(vecs)
        missed += miss * e
        total_energy += e
        comp_modes += 1
        comp_paths += len(group)
    if total_energy <= 1e-14:
        return (1.0, 0.0, comp_modes, comp_paths, all_modes, all_paths)
    return (missed / total_energy, total_energy, comp_modes, comp_paths, all_modes, all_paths)


def evaluate_robust(
    x: np.ndarray,
    *,
    complex_vars: bool,
    eta_balance: float,
    min_f1_energy: float,
    min_d_ratio: float,
    max_l_over_d: float,
    min_second_ratio: float,
    w_vort: float,
    w_vel: float,
    w_rad: float,
    w_l_over_d: float,
) -> RobustMetrics | None:
    F0 = parent_from_params(x, complex_vars=complex_vars)
    fam = induced_fan(F0)
    F1,D,L = fam["F1"], fam["D"], fam["L"]
    if len(F1) < 3:
        return None
    f1_norms = np.array([norm(v) for v in F1.values()], dtype=float)
    f1_energy = float(np.sum(f1_norms**2))
    if f1_energy < min_f1_energy:
        return None
    f1_minmax = float(np.min(f1_norms)/(np.max(f1_norms)+1e-15))
    if f1_minmax < eta_balance:
        return None
    d_energy = float(sum(norm2(v) for v in D.values()))
    l_energy = float(sum(norm2(v) for v in L.values()))
    d_ratio = float(d_energy / ((f1_energy**2)+1e-15))
    if d_ratio < min_d_ratio:
        return None
    l_over_d = float(l_energy/(d_energy+1e-15))
    if l_over_d > max_l_over_d:
        return None
    inters = second_stage_interactions(fam)
    if not inters:
        return None
    vel_miss, coll_e, comp_modes, comp_paths, all_modes, all_paths = comparable_collision_miss(inters, vorticity=False, min_second_ratio=min_second_ratio)
    vort_miss, _, _, _, _, _ = comparable_collision_miss(inters, vorticity=True, min_second_ratio=min_second_ratio)
    e_rad = radius_energy(inters)
    mean_r = sum(r*e for r,e in e_rad.items())/(sum(e_rad.values())+1e-15)
    rad_rel = radius_variance(e_rad)/((mean_r**2)+1e-15)
    if comp_modes == 0:
        # no comparable collisions: treat as an escape only if radius spread is tiny; otherwise it is broad branching.
        pass
    obj = float(w_vort*vort_miss + w_vel*vel_miss + w_rad*rad_rel + w_l_over_d*math.log1p(l_over_d))
    return RobustMetrics(
        objective=obj,
        robust_vel_miss=float(vel_miss),
        robust_vort_miss=float(vort_miss),
        comparable_collision_energy=float(coll_e),
        comparable_collision_modes=int(comp_modes),
        comparable_collision_paths=int(comp_paths),
        all_collision_modes=int(all_modes),
        all_collision_paths=int(all_paths),
        radius_rel_var=float(rad_rel),
        radius_classes=len(e_rad),
        f1_energy=f1_energy,
        f1_minmax=f1_minmax,
        d_energy=d_energy,
        l_energy=l_energy,
        l_over_d=l_over_d,
        d_ratio=d_ratio,
        params=[float(z) for z in np.real(x).tolist()],
    )


def make_obj(args):
    def obj(x: np.ndarray) -> float:
        m = evaluate_robust(
            x,
            complex_vars=args.complex_vars,
            eta_balance=args.eta,
            min_f1_energy=args.min_f1_energy,
            min_d_ratio=args.min_d_ratio,
            max_l_over_d=args.max_l_over_d,
            min_second_ratio=args.min_second_ratio,
            w_vort=args.w_vort,
            w_vel=args.w_vel,
            w_rad=args.w_rad,
            w_l_over_d=args.w_l_over_d,
        )
        if m is None:
            return 1e3
        return m.objective
    return obj


def run_once(args, seed: int) -> dict[str, Any]:
    dim = 12 if args.complex_vars else 6
    bounds = [(-args.bound,args.bound)]*dim
    obj = make_obj(args)
    de = differential_evolution(obj, bounds, seed=seed, maxiter=args.maxiter, popsize=args.popsize, tol=args.tol, polish=False, workers=1, updating="immediate")
    loc = minimize(obj, de.x, method="Nelder-Mead", options={"maxiter":args.local_maxiter,"xatol":1e-10,"fatol":1e-10})
    x = loc.x if loc.fun <= de.fun else de.x
    fun = float(min(loc.fun, de.fun))
    m = evaluate_robust(
        x,
        complex_vars=args.complex_vars,
        eta_balance=args.eta,
        min_f1_energy=args.min_f1_energy,
        min_d_ratio=args.min_d_ratio,
        max_l_over_d=args.max_l_over_d,
        min_second_ratio=args.min_second_ratio,
        w_vort=args.w_vort,
        w_vel=args.w_vel,
        w_rad=args.w_rad,
        w_l_over_d=args.w_l_over_d,
    )
    return {"seed":seed,"fun":fun,"success":bool(de.success),"message":str(de.message),"metrics":None if m is None else asdict(m),"x":[float(v) for v in x.tolist()]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complex", action="store_true", dest="complex_vars")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1200)
    ap.add_argument("--maxiter", type=int, default=100)
    ap.add_argument("--popsize", type=int, default=15)
    ap.add_argument("--local-maxiter", type=int, default=2000)
    ap.add_argument("--bound", type=float, default=3.0)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--eta", type=float, default=0.8)
    ap.add_argument("--min-f1-energy", type=float, default=0.2)
    ap.add_argument("--min-d-ratio", type=float, default=0.3)
    ap.add_argument("--max-l-over-d", type=float, default=5.0)
    ap.add_argument("--min-second-ratio", type=float, default=0.1)
    ap.add_argument("--w-vort", type=float, default=1.0)
    ap.add_argument("--w-vel", type=float, default=0.0)
    ap.add_argument("--w-rad", type=float, default=0.15)
    ap.add_argument("--w-l-over-d", type=float, default=0.0)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    rows = []
    print("Robust first-handoff escape search")
    print(f"complex={args.complex_vars} runs={args.runs} eta={args.eta} min_second_ratio={args.min_second_ratio}")
    for i in range(args.runs):
        row = run_once(args, args.seed+i)
        rows.append(row)
        m = row["metrics"]
        print("-"*100)
        print(f"run={i+1} seed={row['seed']} fun={row['fun']:.10g} success={row['success']}")
        if m is None:
            print("  no metrics")
            continue
        print(f"  robust_vort={m['robust_vort_miss']:.10g} robust_vel={m['robust_vel_miss']:.10g} rad={m['radius_rel_var']:.10g}")
        print(f"  comparable modes={m['comparable_collision_modes']} paths={m['comparable_collision_paths']} energy={m['comparable_collision_energy']:.10g}")
        print(f"  all collision modes={m['all_collision_modes']} paths={m['all_collision_paths']} radius_classes={m['radius_classes']}")
        print(f"  f1minmax={m['f1_minmax']:.10g} f1E={m['f1_energy']:.10g} dE={m['d_energy']:.10g} lE={m['l_energy']:.10g} l/d={m['l_over_d']:.10g}")
        print(f"  x={np.array2string(np.array(row['x']), precision=6, suppress_small=True)}")
    rows.sort(key=lambda r: r["fun"])
    print("="*100)
    print("BEST")
    print(json.dumps(rows[0], indent=2)[:5000])
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json}")

if __name__ == "__main__":
    main()
