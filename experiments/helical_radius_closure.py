#!/usr/bin/env python3
"""Iterate pure-helicity / helical-projected leakage fan closures.

This experiment targets the helical escape found by helical_line_bundle_scan.py:
pure same-helicity assignments nearly eliminate polarization line defect in the first fan.
The question here is whether that escape still pays a radius-fan / heat-detuning tax
under additional closure generations.

We build F1, D, L with pure helical polarizations, compute the second-stage fan using
same interaction families as previous experiments, then iterate all-pairs closure on the
current fan.  At each step we can either keep the full vector output or project each
output mode to its dominant helical component (the most favorable scalar-helicity model).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from navier_stokes_cascade.mode_sets import perm_shell, add_modes
from navier_stokes_cascade.ns_symbol import Mode, Array, ns_bilinear, norm2, norm, squared_radius, Interaction
from navier_stokes_cascade.helical import helical_basis, helical_coeffs, helicity_purity
from navier_stokes_cascade.defects import interactions_between, radius_energy, radius_variance, group_by_output
from experiments.robust_optimize_first_handoff import comparable_collision_miss


def pure_helical(k: Mode, sign: int, amp: complex = 1.0) -> Array:
    hp, hm = helical_basis(k)
    return amp * (hp if sign > 0 else hm)


def dominant_helical_projection(k: Mode, v: Array) -> tuple[Array, int, float]:
    hp, hm = helical_basis(k)
    cp, cm = helical_coeffs(k, v)
    ep, em = abs(cp) ** 2, abs(cm) ** 2
    if ep >= em:
        return cp * hp, +1, ep / (ep + em + 1e-30)
    return cm * hm, -1, em / (ep + em + 1e-30)


def build_initial(sign: int) -> dict[str, Dict[Mode, Array]]:
    return {
        "F1": {k: pure_helical(k, sign) for k in perm_shell((1, 1, 0))},
        "D": {k: pure_helical(k, sign) for k in perm_shell((2, 1, 1))},
        "L": {k: pure_helical(k, sign) for k in perm_shell((2, 1, 0))},
    }


def second_stage_interactions(fam: dict[str, Dict[Mode, Array]]) -> list[Interaction]:
    F1, D, L = fam["F1"], fam["D"], fam["L"]
    inters: list[Interaction] = []
    inters += interactions_between(L, F1)
    inters += interactions_between(D, F1)
    inters += interactions_between(L, L)
    inters += interactions_between(D, D)
    inters += interactions_between(D, L)
    return inters


def all_pair_interactions(active: Dict[Mode, Array], *, include_self_order: bool = True) -> list[Interaction]:
    inters: list[Interaction] = []
    items = list(active.items())
    for i, (p, a) in enumerate(items):
        for j, (q, b) in enumerate(items):
            if not include_self_order and j < i:
                continue
            m = add_modes(p, q)
            if m == (0, 0, 0):
                continue
            out = ns_bilinear(p, a, q, b)
            if norm2(out) > 1e-18:
                inters.append(Interaction(p, q, m, out))
    return inters


def aggregate_outputs(
    inters: list[Interaction],
    *,
    project_helical: bool,
    rel_cut: float,
    max_modes: int,
    normalize: bool,
) -> tuple[Dict[Mode, Array], dict[str, Any]]:
    agg: dict[Mode, Array] = defaultdict(lambda: np.zeros(3, dtype=np.complex128))
    for inter in inters:
        agg[inter.m] += inter.output
    # optional projection to dominant helical line after aggregation per mode
    projected: Dict[Mode, Array] = {}
    purities = []
    signs = {}
    for m, v in agg.items():
        if norm2(v) <= 1e-18:
            continue
        if project_helical:
            vv, s, pur = dominant_helical_projection(m, v)
            projected[m] = vv
            purities.append((norm2(v), pur))
            signs[str(m)] = s
        else:
            projected[m] = v
            purities.append((norm2(v), helicity_purity(m, v)))
    if not projected:
        return {}, {"kept": 0, "signs": signs, "avg_purity": 0.0}
    energies = {m: norm2(v) for m, v in projected.items()}
    max_e = max(energies.values())
    kept_items = [(m, projected[m], energies[m]) for m in projected if energies[m] >= rel_cut * max_e]
    kept_items.sort(key=lambda t: t[2], reverse=True)
    if max_modes and len(kept_items) > max_modes:
        kept_items = kept_items[:max_modes]
    kept = {m: v.copy() for m, v, e in kept_items}
    if normalize and kept:
        total = sum(norm2(v) for v in kept.values())
        if total > 1e-18:
            scale = 1.0 / np.sqrt(total)
            kept = {m: scale * v for m, v in kept.items()}
    avg_purity = sum(e * p for e, p in purities) / (sum(e for e, p in purities) + 1e-30)
    return kept, {"kept": len(kept), "signs": signs, "avg_purity": float(avg_purity), "pre_filter_modes": len(projected)}


def stage_report(name: str, active: Dict[Mode, Array], inters: list[Interaction] | None = None, min_second_ratio: float = 0.1) -> dict[str, Any]:
    energies_by_radius = defaultdict(float)
    dir_cov = np.zeros((3, 3), dtype=float)
    total_active_energy = 0.0
    for k, v in active.items():
        e = norm2(v)
        energies_by_radius[squared_radius(k)] += e
        kk = np.asarray(k, dtype=float)
        nk = np.linalg.norm(kk)
        if nk > 1e-15:
            kh = kk / nk
            dir_cov += e * np.outer(kh, kh)
        total_active_energy += e
    er = dict(sorted(energies_by_radius.items()))
    total = sum(er.values())
    mean = sum(r * e for r, e in er.items()) / (total + 1e-30)
    rel_var = radius_variance(er) / (mean * mean + 1e-30) if er else 0.0
    pur = sum(norm2(v) * helicity_purity(k, v) for k, v in active.items()) / (sum(norm2(v) for v in active.values()) + 1e-30)
    if total_active_energy > 1e-30:
        dir_cov /= total_active_energy
        eigvals, eigvecs = np.linalg.eigh(dir_cov)
        order = np.argsort(eigvals)[::-1]
        dir_eigs = [float(eigvals[i]) for i in order]
        principal = [float(x) for x in eigvecs[:, order[0]].tolist()]
    else:
        dir_eigs = [0.0, 0.0, 0.0]
        principal = [0.0, 0.0, 0.0]
    out: dict[str, Any] = {
        "name": name,
        "modes": len(active),
        "radius_classes": len(er),
        "radius_energy": {str(k): float(v) for k, v in er.items()},
        "radius_min": min(er) if er else None,
        "radius_max": max(er) if er else None,
        "radius_mean": float(mean),
        "radius_rel_var": float(rel_var),
        "direction_cov_eigs": dir_eigs,
        "direction_principal": principal,
        "direction_linearity": float(dir_eigs[0]) if dir_eigs else 0.0,
        "direction_planarity": float(dir_eigs[0] + dir_eigs[1]) if dir_eigs else 0.0,
        "helicity_purity_energy_avg": float(pur),
        "total_energy": float(sum(norm2(v) for v in active.values())),
    }
    if inters is not None:
        vel, ce, cm, cp, am, ap = comparable_collision_miss(inters, vorticity=False, min_second_ratio=min_second_ratio)
        vort, *_ = comparable_collision_miss(inters, vorticity=True, min_second_ratio=min_second_ratio)
        coeff_vals = []
        coeff_weights = []
        for it in inters:
            # Normalize by typical first-order NS scale: |B| / (((|p|+|q|)/2)|a||b|).
            pn = np.linalg.norm(np.asarray(it.p, dtype=float))
            qn = np.linalg.norm(np.asarray(it.q, dtype=float))
            denom = 0.5 * (pn + qn) * norm(active.get(it.p, np.ones(3))) * norm(active.get(it.q, np.ones(3)))
            # If interaction inputs are not from the current active dict (stage1 special families), fall back to output only.
            if denom > 1e-20:
                c = it.output_norm / denom
                coeff_vals.append(c)
                coeff_weights.append(norm2(it.output))
        if coeff_vals:
            coeff_vals_np = np.array(coeff_vals, dtype=float)
            coeff_weights_np = np.array(coeff_weights, dtype=float)
            coeff_mean = float(np.average(coeff_vals_np, weights=coeff_weights_np))
            coeff_min = float(np.min(coeff_vals_np))
            coeff_med = float(np.median(coeff_vals_np))
            coeff_max = float(np.max(coeff_vals_np))
        else:
            coeff_mean = coeff_min = coeff_med = coeff_max = 0.0
        out.update({
            "interaction_count": len(inters),
            "robust_vel_miss": float(vel),
            "robust_vort_miss": float(vort),
            "comparable_collision_energy": float(ce),
            "comparable_collision_modes": int(cm),
            "comparable_collision_paths": int(cp),
            "all_collision_modes": int(am),
            "all_collision_paths": int(ap),
            "coeff_norm_energy_mean": coeff_mean,
            "coeff_norm_min": coeff_min,
            "coeff_norm_median": coeff_med,
            "coeff_norm_max": coeff_max,
        })
    return out


def run(args) -> dict[str, Any]:
    fam = build_initial(args.sign)
    active0: Dict[Mode, Array] = {}
    for d in fam.values():
        active0.update(d)
    stages = [stage_report("initial_F1_D_L", active0, None, args.min_second_ratio)]

    inters = second_stage_interactions(fam)
    active, meta = aggregate_outputs(
        inters,
        project_helical=args.project_helical,
        rel_cut=args.rel_cut,
        max_modes=args.max_modes,
        normalize=args.normalize,
    )
    rep = stage_report("stage1_second_stage_outputs", active, inters, args.min_second_ratio)
    rep["aggregate_meta"] = meta
    stages.append(rep)

    current = active
    for n in range(2, args.depth + 1):
        inters = all_pair_interactions(current, include_self_order=True)
        current, meta = aggregate_outputs(
            inters,
            project_helical=args.project_helical,
            rel_cut=args.rel_cut,
            max_modes=args.max_modes,
            normalize=args.normalize,
        )
        rep = stage_report(f"stage{n}_self_closure", current, inters, args.min_second_ratio)
        rep["aggregate_meta"] = meta
        stages.append(rep)
        if not current:
            break
    return {
        "args": {k: str(v) if hasattr(v, "as_posix") else v for k, v in vars(args).items()},
        "stages": stages,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sign", type=int, choices=[-1, 1], default=-1)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--project-helical", action="store_true")
    ap.add_argument("--rel-cut", type=float, default=1e-6)
    ap.add_argument("--max-modes", type=int, default=250)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--min-second-ratio", type=float, default=0.1)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    res = run(args)
    print("Helical radius closure experiment")
    print(f"sign={args.sign} project_helical={args.project_helical} depth={args.depth} rel_cut={args.rel_cut} max_modes={args.max_modes} normalize={args.normalize}")
    for st in res["stages"]:
        print("-" * 110)
        print(f"{st['name']}: modes={st['modes']} radii={st['radius_classes']} range=({st['radius_min']},{st['radius_max']}) rel_var={st['radius_rel_var']:.8g} purity={st['helicity_purity_energy_avg']:.8g} dir_eigs={st['direction_cov_eigs']}")
        if "interaction_count" in st:
            print(f"  inters={st['interaction_count']} robust_vort={st['robust_vort_miss']:.8g} robust_vel={st['robust_vel_miss']:.8g} comp_modes={st['comparable_collision_modes']} comp_paths={st['comparable_collision_paths']} coeff_mean={st['coeff_norm_energy_mean']:.8g} coeff_med={st['coeff_norm_median']:.8g}")
        print(f"  radius_energy={st['radius_energy']}")
        if "aggregate_meta" in st:
            print(f"  aggregate_meta={{'pre_filter_modes': {st['aggregate_meta'].get('pre_filter_modes')}, 'kept': {st['aggregate_meta'].get('kept')}, 'avg_purity': {st['aggregate_meta'].get('avg_purity'):.8g}}}")
    if args.json:
        args.json.write_text(json.dumps(res, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
