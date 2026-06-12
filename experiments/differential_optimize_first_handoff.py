#!/usr/bin/env python3
"""Differential-evolution search for first-handoff escape geometries.

This is a stronger follow-up to optimize_first_handoff.py.  It searches for
parent amplitudes on the first shell that minimize the proposed obstruction
score while satisfying nondegenerate first-handoff constraints.

The point is adversarial: find low radius/polarization/vorticity defect cases.
If the optimizer can drive the score toward zero while keeping healthy desired
transfer, our proposed universal defect theorem is false or needs sharpening.
If it bottoms out at a positive value, that gives evidence for a finite lemma.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution, minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.optimize_first_handoff import evaluate, SampleMetrics


def constraint_penalty(
    m: SampleMetrics | None,
    *,
    eta: float,
    min_f1_energy: float,
    min_d_ratio: float,
    max_l_over_d: float,
    hard_fail: float = 1e3,
) -> float:
    if m is None:
        return hard_fail
    pen = 0.0
    # These are mostly redundant with evaluate rejection, but useful when we use a lax
    # evaluate then penalize continuously in future variants.
    if m.f1_minmax < eta:
        pen += 100.0 * (eta - m.f1_minmax) ** 2
    if m.f1_energy < min_f1_energy:
        pen += 100.0 * (min_f1_energy - m.f1_energy) ** 2
    d_ratio = m.d_energy / ((m.f1_energy ** 2) + 1e-15)
    if d_ratio < min_d_ratio:
        pen += 100.0 * (min_d_ratio - d_ratio) ** 2
    if m.l_over_d > max_l_over_d:
        pen += 10.0 * (m.l_over_d - max_l_over_d) ** 2
    return pen


def make_objective(args: argparse.Namespace):
    def obj(x: np.ndarray) -> float:
        m = evaluate(
            x,
            idx=0,
            complex_vars=args.complex_vars,
            eta_balance=args.eta,
            min_f1_energy=args.min_f1_energy,
            min_d_ratio=args.min_d_ratio,
            max_l_over_d=args.max_l_over_d,
        )
        if m is None:
            return 1e3
        score = args.w_vort * m.vort_miss + args.w_vel * m.vel_miss + args.w_rad * m.radius_rel_var
        # Optional mild preference against very leakage-dominated configurations even below hard cap.
        if args.w_l_over_d:
            score += args.w_l_over_d * math.log1p(m.l_over_d)
        return float(score)
    return obj


def metric_for_x(x: np.ndarray, args: argparse.Namespace) -> SampleMetrics | None:
    return evaluate(
        x,
        idx=0,
        complex_vars=args.complex_vars,
        eta_balance=args.eta,
        min_f1_energy=args.min_f1_energy,
        min_d_ratio=args.min_d_ratio,
        max_l_over_d=args.max_l_over_d,
    )


def run_once(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    dim = 12 if args.complex_vars else 6
    bounds = [(-args.bound, args.bound)] * dim
    obj = make_objective(args)
    result = differential_evolution(
        obj,
        bounds=bounds,
        seed=seed,
        maxiter=args.maxiter,
        popsize=args.popsize,
        tol=args.tol,
        polish=False,
        updating="immediate",
        workers=1,
        disp=False,
    )
    x = result.x
    local = minimize(obj, x, method="Nelder-Mead", options={"maxiter": args.local_maxiter, "xatol": 1e-9, "fatol": 1e-9})
    if local.fun <= result.fun:
        x = local.x
        fun = float(local.fun)
        opt_kind = "de+nelder-mead"
    else:
        fun = float(result.fun)
        opt_kind = "de"
    m = metric_for_x(x, args)
    return {
        "seed": seed,
        "optimizer": opt_kind,
        "fun": fun,
        "success": bool(result.success),
        "de_message": str(result.message),
        "metrics": asdict(m) if m is not None else None,
        "x": [float(v) for v in x.tolist()],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--complex", action="store_true", dest="complex_vars")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--popsize", type=int, default=15)
    ap.add_argument("--local-maxiter", type=int, default=1500)
    ap.add_argument("--bound", type=float, default=3.0)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--eta", type=float, default=0.8)
    ap.add_argument("--min-f1-energy", type=float, default=0.2)
    ap.add_argument("--min-d-ratio", type=float, default=0.3)
    ap.add_argument("--max-l-over-d", type=float, default=5.0)
    ap.add_argument("--w-vort", type=float, default=1.0)
    ap.add_argument("--w-vel", type=float, default=0.0)
    ap.add_argument("--w-rad", type=float, default=0.15)
    ap.add_argument("--w-l-over-d", type=float, default=0.0)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    print("Differential evolution first-handoff escape search")
    print(f"complex={args.complex_vars} runs={args.runs} eta={args.eta} min_f1={args.min_f1_energy} min_d_ratio={args.min_d_ratio} max_l_over_d={args.max_l_over_d}")
    print(f"weights: vort={args.w_vort} vel={args.w_vel} rad={args.w_rad} l_over_d={args.w_l_over_d}")
    for r in range(args.runs):
        row = run_once(args, args.seed + r)
        rows.append(row)
        m = row["metrics"]
        print("-"*100)
        print(f"run={r+1} seed={row['seed']} optimizer={row['optimizer']} fun={row['fun']:.8g} success={row['success']}")
        if m is None:
            print("  no accepted metrics")
            continue
        print(f"  objective(metrics)={m['objective']:.8g} vel_miss={m['vel_miss']:.8g} vort_miss={m['vort_miss']:.8g}")
        print(f"  radius_rel_var={m['radius_rel_var']:.8g} radius_classes={m['radius_classes']} collision_modes={m['collision_modes']} collision_paths={m['collision_paths']}")
        print(f"  f1_energy={m['f1_energy']:.8g} f1_minmax={m['f1_minmax']:.8g} d_energy={m['d_energy']:.8g} l_energy={m['l_energy']:.8g} l_over_d={m['l_over_d']:.8g}")
        print(f"  x={np.array2string(np.array(row['x']), precision=6, suppress_small=True)}")
    rows.sort(key=lambda z: z["fun"])
    print("="*100)
    print("BEST")
    best = rows[0]
    print(json.dumps(best, indent=2)[:4000])
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
