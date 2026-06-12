#!/usr/bin/env python3
"""Scan pure-helicity line assignments on the first leakage fan.

For every assignment sigma(k) in {+,-} on F1 ∪ D ∪ L, assign the pure helical
polarization h_sigma(k) and compute the same second-stage interaction fan.
This asks whether a pure-helicity complex line bundle can evade the collision
polarization/vorticity defects seen in the real symmetric induced amplitudes.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from navier_stokes_cascade.mode_sets import perm_shell
from navier_stokes_cascade.helical import helical_basis, helicity_purity
from navier_stokes_cascade.defects import interactions_between, radius_energy, radius_variance, group_by_output
from navier_stokes_cascade.ns_symbol import Mode, Array, norm2, one_line_miss_fraction, vorticity_polarization
from experiments.robust_optimize_first_handoff import comparable_collision_miss


def pure_helical(k: Mode, sign: int) -> Array:
    hp, hm = helical_basis(k)
    return hp if sign > 0 else hm


def build_families(signs: dict[Mode, int]) -> dict[str, Dict[Mode, Array]]:
    F1_modes = perm_shell((1,1,0))
    D_modes = perm_shell((2,1,1))
    L_modes = perm_shell((2,1,0))
    return {
        "F1": {k: pure_helical(k, signs[k]) for k in F1_modes},
        "D": {k: pure_helical(k, signs[k]) for k in D_modes},
        "L": {k: pure_helical(k, signs[k]) for k in L_modes},
    }


def second_stage(fam):
    F1,D,L = fam["F1"],fam["D"],fam["L"]
    inters=[]
    inters += interactions_between(L,F1)
    inters += interactions_between(D,F1)
    inters += interactions_between(L,L)
    inters += interactions_between(D,D)
    inters += interactions_between(D,L)
    return inters


def metric_for_signs(modes: list[Mode], bitmask: int, min_second_ratio: float) -> dict[str, Any]:
    signs={}
    for i,k in enumerate(modes):
        signs[k]=1 if ((bitmask>>i)&1) else -1
    fam=build_families(signs)
    inters=second_stage(fam)
    vel, ce, cm, cp, am, ap = comparable_collision_miss(inters, vorticity=False, min_second_ratio=min_second_ratio)
    vort, *_ = comparable_collision_miss(inters, vorticity=True, min_second_ratio=min_second_ratio)
    er=radius_energy(inters)
    mean=sum(r*e for r,e in er.items())/(sum(er.values())+1e-15)
    rad=radius_variance(er)/((mean*mean)+1e-15)
    total=sum(norm2(x.output) for x in inters)
    # helicity purity of outputs: how much the output fan remains pure helical mode-by-mode.
    pur_num=0.0; pur_den=0.0
    for x in inters:
        e=norm2(x.output)
        pur_num += e*helicity_purity(x.m, x.output)
        pur_den += e
    purity=pur_num/(pur_den+1e-15)
    return {
        "bitmask": bitmask,
        "objective": float(vort + 0.15*rad),
        "robust_vel_miss": float(vel),
        "robust_vort_miss": float(vort),
        "radius_rel_var": float(rad),
        "radius_classes": len(er),
        "total_output_energy": float(total),
        "comparable_collision_energy": float(ce),
        "comparable_collision_modes": int(cm),
        "comparable_collision_paths": int(cp),
        "all_collision_modes": int(am),
        "all_collision_paths": int(ap),
        "output_helicity_purity_energy_avg": float(purity),
        "signs": {str(k): signs[k] for k in modes},
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--min-second-ratio", type=float, default=0.1)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--json", type=Path, default=None)
    args=ap.parse_args()
    modes=sorted(set(perm_shell((1,1,0))+perm_shell((2,1,1))+perm_shell((2,1,0))))
    rows=[]
    for bitmask in range(1<<len(modes)):
        rows.append(metric_for_signs(modes, bitmask, args.min_second_ratio))
    rows.sort(key=lambda r: r["objective"])
    print("Pure-helicity first-fan line-bundle scan")
    print(f"modes={len(modes)} assignments={len(rows)} min_second_ratio={args.min_second_ratio}")
    for i,r in enumerate(rows[:args.top],1):
        print("-"*100)
        print(f"rank={i} obj={r['objective']:.10g} robust_vort={r['robust_vort_miss']:.10g} robust_vel={r['robust_vel_miss']:.10g} rad={r['radius_rel_var']:.10g}")
        print(f"  comp_modes={r['comparable_collision_modes']} comp_paths={r['comparable_collision_paths']} all_modes={r['all_collision_modes']} all_paths={r['all_collision_paths']} radius_classes={r['radius_classes']}")
        print(f"  output_helicity_purity_avg={r['output_helicity_purity_energy_avg']:.10g} totalE={r['total_output_energy']:.10g}")
        print(f"  bitmask={r['bitmask']} signs={r['signs']}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json}")

if __name__ == "__main__":
    main()
