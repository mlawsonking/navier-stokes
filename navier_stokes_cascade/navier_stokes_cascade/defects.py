"""Defect/diagnostic metrics for finite Fourier cascade fans."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .mode_sets import add_modes
from .ns_symbol import (
    Array,
    Interaction,
    Mode,
    angle_between,
    ns_bilinear,
    one_line_miss_fraction,
    squared_radius,
    vorticity_polarization,
)


@dataclass
class CollisionReport:
    mode: Mode
    radius2: int
    interactions: List[Interaction]
    velocity_miss_fraction: float
    vorticity_miss_fraction: float
    max_velocity_angle_deg: float
    max_vorticity_angle_deg: float


def interactions_between(A: Dict[Mode, Array], B: Dict[Mode, Array]) -> list[Interaction]:
    interactions: list[Interaction] = []
    for p, a in A.items():
        for q, b in B.items():
            m = add_modes(p, q)
            out = ns_bilinear(p, a, q, b)
            if np.linalg.norm(out) > 1e-10:
                interactions.append(Interaction(p=p, q=q, m=m, output=out))
    return interactions


def group_by_output(interactions: Iterable[Interaction]) -> Dict[Mode, List[Interaction]]:
    g: Dict[Mode, List[Interaction]] = defaultdict(list)
    for inter in interactions:
        g[inter.m].append(inter)
    return dict(g)


def collision_reports(interactions: Iterable[Interaction], *, min_paths: int = 2) -> list[CollisionReport]:
    reports: list[CollisionReport] = []
    for m, group in group_by_output(interactions).items():
        if len(group) < min_paths:
            continue
        outs = [x.output for x in group]
        vorts = [vorticity_polarization(m, x.output) for x in group]
        vel_angles = []
        vort_angles = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                vel_angles.append(np.degrees(angle_between(outs[i], outs[j])))
                vort_angles.append(np.degrees(angle_between(vorts[i], vorts[j])))
        reports.append(
            CollisionReport(
                mode=m,
                radius2=squared_radius(m),
                interactions=group,
                velocity_miss_fraction=one_line_miss_fraction(outs),
                vorticity_miss_fraction=one_line_miss_fraction(vorts),
                max_velocity_angle_deg=max(vel_angles) if vel_angles else 0.0,
                max_vorticity_angle_deg=max(vort_angles) if vort_angles else 0.0,
            )
        )
    return sorted(reports, key=lambda r: (r.radius2, r.mode))


def radius_energy(interactions: Iterable[Interaction]) -> dict[int, float]:
    e: dict[int, float] = defaultdict(float)
    for inter in interactions:
        e[inter.radius2] += inter.output_norm ** 2
    return dict(sorted(e.items()))


def radius_variance(energy_by_radius: dict[int, float]) -> float:
    total = sum(energy_by_radius.values())
    if total <= 1e-14:
        return 0.0
    mean = sum(r * e for r, e in energy_by_radius.items()) / total
    return sum(((r - mean) ** 2) * e for r, e in energy_by_radius.items()) / total


def summarize_collision(report: CollisionReport) -> str:
    lines = []
    lines.append(f"mode={report.mode} |k|^2={report.radius2} paths={len(report.interactions)}")
    lines.append(f"  vel miss={report.velocity_miss_fraction:.6f}, max angle={report.max_velocity_angle_deg:.3f} deg")
    lines.append(f"  vort miss={report.vorticity_miss_fraction:.6f}, max angle={report.max_vorticity_angle_deg:.3f} deg")
    for inter in report.interactions:
        out = np.real_if_close(inter.output, tol=1000)
        lines.append(f"    {inter.p}+{inter.q}->{inter.m}, |out|={inter.output_norm:.6g}, out={np.array2string(out, precision=6)}")
    return "\n".join(lines)
