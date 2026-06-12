#!/usr/bin/env python3
"""Parameter sweep for helical closure diagnostics.

This checks whether the pure-helicity near-ray collapse is robust to truncation choices
(relative energy cutoff and maximum kept modes), and contrasts projected-helicity closure
against full-vector closure.
"""
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.helical_radius_closure import run


def make_args(project, rel_cut, max_modes, depth, sign=-1):
    return Namespace(
        sign=sign,
        depth=depth,
        project_helical=project,
        rel_cut=rel_cut,
        max_modes=max_modes,
        normalize=True,
        min_second_ratio=0.1,
        json=None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    configs = [
        (True, 1e-3, 0, 4),
        (True, 1e-4, 0, 4),
        (True, 1e-5, 1000, 4),
        (True, 1e-6, 250, 5),
        (False, 1e-6, 250, 4),
    ]
    all_results = []
    print("Helical closure parameter sweep")
    print("project rel_cut max_modes depth stage modes radii linearity purity robust_vort coeff_mean")
    for project, rel_cut, max_modes, depth in configs:
        res = run(make_args(project, rel_cut, max_modes, depth))
        compact = {
            "project_helical": project,
            "rel_cut": rel_cut,
            "max_modes": max_modes,
            "depth": depth,
            "stages": [],
        }
        for st in res["stages"]:
            row = {
                "name": st["name"],
                "modes": st["modes"],
                "radius_classes": st["radius_classes"],
                "radius_rel_var": st["radius_rel_var"],
                "direction_linearity": st["direction_linearity"],
                "helicity_purity": st["helicity_purity_energy_avg"],
                "robust_vort_miss": st.get("robust_vort_miss"),
                "coeff_norm_energy_mean": st.get("coeff_norm_energy_mean"),
            }
            compact["stages"].append(row)
            if st["name"] != "initial_F1_D_L":
                print(
                    f"{int(project)} {rel_cut:g} {max_modes:4d} {depth:2d} {st['name']:28s} "
                    f"{st['modes']:5d} {st['radius_classes']:5d} {st['direction_linearity']:.6f} "
                    f"{st['helicity_purity_energy_avg']:.6f} {st.get('robust_vort_miss', 0):.6g} "
                    f"{st.get('coeff_norm_energy_mean', 0):.6g}"
                )
        all_results.append(compact)
    if args.json:
        args.json.write_text(json.dumps(all_results, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
