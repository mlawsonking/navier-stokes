#!/usr/bin/env python3
"""Coefficient depletion for near-ray helical interactions.

The helical closure experiment suggests the pure-helicity escape collapses toward a ray.
This script tests the basic scaling mechanism directly: for modes p,q with bounded
integer defects from the same ray, compute the normalized NS coefficient

    |B(p,a;q,b)| / (N_typ |a||b|)

for helical polarizations.  Nondegenerate interactions have O(1) normalized
coefficient.  Near-parallel ray interactions should decay like an inverse scale.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from navier_stokes_cascade.helical import helical_basis
from navier_stokes_cascade.ns_symbol import ns_bilinear, norm, norm2, angle_between


def pure_helical(k: tuple[int, int, int], sign: int):
    hp, hm = helical_basis(k)
    return hp if sign > 0 else hm


def coeff_norm(p, q, sp, sq):
    a = pure_helical(p, sp)
    b = pure_helical(q, sq)
    out = ns_bilinear(p, a, q, b)
    ntyp = 0.5 * (np.linalg.norm(np.asarray(p, dtype=float)) + np.linalg.norm(np.asarray(q, dtype=float)))
    return norm(out) / (ntyp * norm(a) * norm(b) + 1e-30)


def run_family(name: str, Ns: Iterable[int]):
    rows = []
    for N in Ns:
        if name == "minus_pair":
            p = (N, N, N - 1)
            q = (N, N - 1, N)
        elif name == "plus_minus_pair":
            p = (N, N, N - 1)
            q = (N, N, N + 1)
        elif name == "two_defects":
            p = (N + 1, N, N - 1)
            q = (N, N - 1, N + 1)
        elif name == "transverse_control":
            p = (N, 0, 0)
            q = (0, N, 0)
        else:
            raise ValueError(name)
        angle = angle_between(np.asarray(p, dtype=np.complex128), np.asarray(q, dtype=np.complex128))
        row = {
            "N": N,
            "p": p,
            "q": q,
            "angle_deg": float(np.degrees(angle)),
            "sin_angle": float(np.sin(angle)),
        }
        vals = {}
        for sp in (-1, 1):
            for sq in (-1, 1):
                key = f"{sp:+d}{sq:+d}"
                vals[key] = float(coeff_norm(p, q, sp, sq))
        row["coeffs"] = vals
        row["max_coeff"] = max(vals.values())
        row["min_coeff"] = min(vals.values())
        rows.append(row)
    return rows


def fit_power(rows, field="max_coeff"):
    xs = np.log(np.array([r["N"] for r in rows], dtype=float))
    ys = np.log(np.array([max(r[field], 1e-300) for r in rows], dtype=float))
    A = np.vstack([xs, np.ones_like(xs)]).T
    slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]
    return float(slope), float(np.exp(intercept))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    Ns = [2, 3, 4, 5, 8, 13, 21, 34, 55, 89, 144]
    families = ["minus_pair", "plus_minus_pair", "two_defects", "transverse_control"]
    result = {fam: run_family(fam, Ns) for fam in families}
    print("Near-ray coefficient depletion scan")
    for fam, rows in result.items():
        slope, pref = fit_power(rows, "max_coeff")
        print("-" * 100)
        print(f"{fam}: max_coeff ~ {pref:.6g} * N^{slope:.3f}")
        print("N angle_deg max_coeff coeffs")
        for r in rows:
            coeff_str = ", ".join(f"{k}:{v:.3e}" for k, v in r["coeffs"].items())
            print(f"{r['N']:4d} {r['angle_deg']:10.6f} {r['max_coeff']:.6e} {coeff_str}")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
