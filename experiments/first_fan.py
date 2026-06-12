#!/usr/bin/env python3
"""First leakage fan experiment.

Computes:
- symmetric F0/F1/D/L amplitudes
- second-stage interaction radius energy
- all output-mode collisions and line-bundle/vorticity defects
- explicit checks for the known (3,1,2) and (3,3,3) split examples
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from navier_stokes_cascade.defects import (
    collision_reports,
    interactions_between,
    radius_energy,
    radius_variance,
    summarize_collision,
)
from navier_stokes_cascade.helical import helicity_purity, helical_coeffs
from navier_stokes_cascade.mode_sets import first_fan_amplitudes
from navier_stokes_cascade.ns_symbol import angle_between, ns_bilinear, norm, real_vec, vorticity_polarization


def print_section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def main() -> None:
    fam = first_fan_amplitudes()
    F1, D, L = fam["F1"], fam["D"], fam["L"]

    print_section("First fan amplitudes")
    for name, d in [("F1", F1), ("D", D), ("L", L)]:
        print(f"{name}:")
        for k, a in sorted(d.items()):
            print(f"  {k}: {np.array2string(np.real_if_close(a), precision=6)} |a|={norm(a):.6g}")

    print_section("Second-stage interactions and radius energy")
    blocks = {
        "LxF1": interactions_between(L, F1),
        "DxF1": interactions_between(D, F1),
        "LxL": interactions_between(L, L),
        "DxD": interactions_between(D, D),
        "DxL": interactions_between(D, L),
    }
    all_interactions = []
    for name, inters in blocks.items():
        all_interactions.extend(inters)
        e = radius_energy(inters)
        print(f"{name}: radii={list(e.keys())}, energy_by_radius={{{', '.join(f'{k}: {v:.6g}' for k, v in e.items())}}}")
    e_all = radius_energy(all_interactions)
    print(f"ALL radii={list(e_all.keys())}")
    print(f"ALL energy_by_radius={{{', '.join(f'{k}: {v:.6g}' for k, v in e_all.items())}}}")
    print(f"ALL radius variance={radius_variance(e_all):.6g}")

    print_section("Collision reports: all second-stage blocks combined")
    reports = collision_reports(all_interactions, min_paths=2)
    print(f"collision modes with >=2 paths: {len(reports)}")
    # Sort by vorticity miss descending to surface most serious line-bundle defects.
    reports_sorted = sorted(reports, key=lambda r: r.vorticity_miss_fraction, reverse=True)
    for r in reports_sorted[:25]:
        print(summarize_collision(r))

    print_section("Explicit split: m=(3,1,2)")
    pA, qA = (2, 1, 1), (1, 0, 1)
    aA, bA = D[pA], F1[qA]
    vA = ns_bilinear(pA, aA, qA, bA)
    pB, qB = (2, 1, 0), (1, 0, 2)
    aB, bB = L[pB], L[qB]
    vB = ns_bilinear(pB, aB, qB, bB)
    m = (3, 1, 2)
    print(f"vA={np.array2string(np.real_if_close(vA), precision=6)}, |vA|={norm(vA):.6g}")
    print(f"vB={np.array2string(np.real_if_close(vB), precision=6)}, |vB|={norm(vB):.6g}")
    print(f"velocity angle deg={np.degrees(angle_between(vA, vB)):.6g}")
    zA, zB = vorticity_polarization(m, vA), vorticity_polarization(m, vB)
    print(f"zA={np.array2string(np.real_if_close(zA), precision=6)}")
    print(f"zB={np.array2string(np.real_if_close(zB), precision=6)}")
    print(f"vorticity angle deg={np.degrees(angle_between(zA, zB)):.6g}")
    print(f"helicity purity vA={helicity_purity(m, vA):.6g}, vB={helicity_purity(m, vB):.6g}")
    print(f"helical coeffs vA={helical_coeffs(m, vA)}")
    print(f"helical coeffs vB={helical_coeffs(m, vB)}")

    print_section("Explicit descendant split: k=(3,3,3)")
    # Use the exact branch from prior reasoning.
    q = (0, 2, 1)
    b = L[q]
    wA = ns_bilinear(m, vA, q, b)
    wB = ns_bilinear(m, vB, q, b)
    k = (3, 3, 3)
    print(f"wA={np.array2string(np.real_if_close(wA), precision=6)}, |wA|={norm(wA):.6g}")
    print(f"wB={np.array2string(np.real_if_close(wB), precision=6)}, |wB|={norm(wB):.6g}")
    print(f"velocity angle deg={np.degrees(angle_between(wA, wB)):.6g}")
    oA, oB = vorticity_polarization(k, wA), vorticity_polarization(k, wB)
    print(f"omegaA={np.array2string(np.real_if_close(oA), precision=6)}")
    print(f"omegaB={np.array2string(np.real_if_close(oB), precision=6)}")
    print(f"vorticity angle deg={np.degrees(angle_between(oA, oB)):.6g}")
    print(f"helicity purity wA={helicity_purity(k, wA):.6g}, wB={helicity_purity(k, wB):.6g}")
    print(f"helical coeffs wA={helical_coeffs(k, wA)}")
    print(f"helical coeffs wB={helical_coeffs(k, wB)}")


if __name__ == "__main__":
    main()
