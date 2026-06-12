#!/usr/bin/env python3
"""Classify low-defect first-handoff optimizer minima.

The differential optimizer repeatedly produced low-defect samples near one of the two
3-cycle off-diagonal parent-polarization patterns:

    cycle + : e1->e2, e2->e3, e3->e1  (A,D,E dominant)
    cycle - : e1->e3, e2->e1, e3->e2  (B,C,F dominant)

This script loads optimizer JSON outputs and reports how concentrated each sample is
near those cycles.  It also evaluates exact cycle-family slices to see whether the
apparent minimizer is a boundary artifact or a real finite obstruction family.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution, minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.optimize_first_handoff import evaluate, parent_from_params, induced_fan
from navier_stokes_cascade.ns_symbol import norm2, norm

# Parameter order for real variables: A,B,C,D,E,F
CYCLE_PLUS = [0, 3, 4]   # A,D,E: e1->e2, e2->e3, e3->e1
CYCLE_MINUS = [1, 2, 5]  # B,C,F: e1->e3, e2->e1, e3->e2


def split_params(x: np.ndarray, complex_vars: bool) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if complex_vars:
        return x[:6] + 1j * x[6:12]
    return x[:6].astype(np.complex128)


def cycle_concentration(x: np.ndarray, complex_vars: bool) -> dict[str, float | str]:
    z = split_params(x, complex_vars)
    total = float(np.sum(np.abs(z) ** 2))
    if total <= 1e-15:
        return {"plus": 0.0, "minus": 0.0, "best": "none", "off_cycle": 1.0}
    plus = float(np.sum(np.abs(z[CYCLE_PLUS]) ** 2) / total)
    minus = float(np.sum(np.abs(z[CYCLE_MINUS]) ** 2) / total)
    best = "plus" if plus >= minus else "minus"
    return {"plus": plus, "minus": minus, "best": best, "off_cycle": 1.0 - max(plus, minus)}


def child_values(x: np.ndarray, complex_vars: bool) -> dict[str, Any]:
    A,B,C,D,E,F = split_params(x, complex_vars)
    X = A*D + B*C
    Y = B*F + E*A
    Z = D*E + F*C
    vals = np.array([X,Y,Z], dtype=np.complex128)
    n = np.abs(vals)
    return {
        "X": [float(np.real(X)), float(np.imag(X))],
        "Y": [float(np.real(Y)), float(np.imag(Y))],
        "Z": [float(np.real(Z)), float(np.imag(Z))],
        "abs": [float(v) for v in n.tolist()],
        "minmax": float(np.min(n)/(np.max(n)+1e-15)),
    }


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in paths:
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        for r in data:
            rr = dict(r)
            rr["source"] = str(p)
            rows.append(rr)
    return rows


def report_json(paths: list[Path], complex_vars_hint: bool | None = None) -> None:
    rows = read_rows(paths)
    print(f"loaded rows={len(rows)}")
    for r in sorted(rows, key=lambda z: z.get("fun", 1e99)):
        if r.get("metrics") is None:
            continue
        x = np.array(r["x"], dtype=float)
        complex_vars = complex_vars_hint if complex_vars_hint is not None else (len(x) == 12)
        c = cycle_concentration(x, complex_vars)
        ch = child_values(x, complex_vars)
        m = r["metrics"]
        print("-"*100)
        print(f"source={r['source']} seed={r.get('seed')} fun={r.get('fun'):.8g} complex={complex_vars}")
        
        vort = m.get('vort_miss', m.get('robust_vort_miss', float('nan')))
        vel = m.get('vel_miss', m.get('robust_vel_miss', float('nan')))
        print(f"  metrics: vort={vort:.8g} vel={vel:.8g} rad={m['radius_rel_var']:.8g} l/d={m['l_over_d']:.8g} f1minmax={m['f1_minmax']:.8g}")
        print(f"  cycle: best={c['best']} plus={c['plus']:.8g} minus={c['minus']:.8g} off_cycle={c['off_cycle']:.8g}")
        print(f"  child abs={np.array2string(np.array(ch['abs']), precision=6)} child_minmax={ch['minmax']:.8g}")
        print(f"  x={np.array2string(x, precision=6, suppress_small=True)}")


def cycle_params(u: np.ndarray, family: str) -> np.ndarray:
    """Return real 6-vector on exact plus/minus 3-cycle family.

    u are the three real entries on the chosen cycle. We rely on evaluate's
    normalization, so no normalization is done here.
    """
    x = np.zeros(6, dtype=float)
    idx = CYCLE_PLUS if family == "plus" else CYCLE_MINUS
    x[idx] = u
    return x


def scan_cycle_family(family: str, eta: float, min_f1_energy: float, min_d_ratio: float, max_l_over_d: float) -> dict[str, Any]:
    def obj_y(y: np.ndarray) -> float:
        # exponential variables keep magnitudes positive; signs are handled separately outside.
        x = cycle_params(y, family)
        m = evaluate(x, complex_vars=False, eta_balance=eta, min_f1_energy=min_f1_energy, min_d_ratio=min_d_ratio, max_l_over_d=max_l_over_d)
        if m is None:
            return 1e3
        return float(m.vort_miss + 0.15*m.radius_rel_var)

    best: dict[str, Any] | None = None
    # Check sign patterns too; magnitudes optimized by DE.
    for signs in [(sx,sy,sz) for sx in [1,-1] for sy in [1,-1] for sz in [1,-1]]:
        def obj_log(t: np.ndarray, signs=signs) -> float:
            mags = np.exp(t)
            y = mags * np.array(signs, dtype=float)
            return obj_y(y)
        res = differential_evolution(obj_log, [(-3,3)]*3, seed=100 + hash((family, signs)) % 10000, maxiter=70, popsize=12, polish=False, tol=1e-8)
        loc = minimize(obj_log, res.x, method="Nelder-Mead", options={"maxiter": 1000, "xatol":1e-10, "fatol":1e-10})
        t = loc.x if loc.fun <= res.fun else res.x
        fun = float(min(loc.fun, res.fun))
        mags = np.exp(t)
        y = mags * np.array(signs, dtype=float)
        x = cycle_params(y, family)
        m = evaluate(x, complex_vars=False, eta_balance=eta, min_f1_energy=min_f1_energy, min_d_ratio=min_d_ratio, max_l_over_d=max_l_over_d)
        rec = {"family": family, "signs": signs, "fun": fun, "x": x.tolist(), "metrics": None if m is None else m.__dict__}
        if best is None or fun < best["fun"]:
            best = rec
    assert best is not None
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_files", nargs="*", type=Path)
    ap.add_argument("--cycle-scan", action="store_true")
    ap.add_argument("--eta", type=float, default=0.8)
    ap.add_argument("--min-f1-energy", type=float, default=0.2)
    ap.add_argument("--min-d-ratio", type=float, default=0.3)
    ap.add_argument("--max-l-over-d", type=float, default=5.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.json_files:
        report_json(args.json_files)
    results = []
    if args.cycle_scan:
        print("="*100)
        print("Exact 3-cycle family optimization")
        for fam in ["plus", "minus"]:
            rec = scan_cycle_family(fam, args.eta, args.min_f1_energy, args.min_d_ratio, args.max_l_over_d)
            results.append(rec)
            m = rec["metrics"]
            print("-"*100)
            print(f"family={fam} signs={rec['signs']} fun={rec['fun']:.10g}")
            if m:
                print(f"  vort={m['vort_miss']:.10g} vel={m['vel_miss']:.10g} rad={m['radius_rel_var']:.10g} l/d={m['l_over_d']:.10g} f1minmax={m['f1_minmax']:.10g}")
                print(f"  f1_energy={m['f1_energy']:.10g} d_energy={m['d_energy']:.10g} l_energy={m['l_energy']:.10g}")
            print(f"  x={np.array2string(np.array(rec['x']), precision=8, suppress_small=True)}")
    if args.out:
        args.out.write_text(json.dumps(results, indent=2, default=str))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
