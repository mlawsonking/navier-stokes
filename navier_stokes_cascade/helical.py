"""Helical-basis utilities for future experiments.

The helical basis h_+(k), h_-(k) satisfies i k x h_sigma = sigma |k| h_sigma.
The implementation chooses a deterministic real reference vector not parallel to k.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from .ns_symbol import Array, hdot, mode, norm


def orthonormal_frame_perp(k: Iterable[int] | Array) -> Tuple[Array, Array]:
    kk = mode(k) if not isinstance(k, np.ndarray) else k.astype(np.complex128)
    kr = np.real(kk).astype(float)
    candidates = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    ref = min(candidates, key=lambda e: abs(float(np.dot(e, kr))) / (np.linalg.norm(kr) + 1e-15))
    e1 = ref.astype(np.complex128) - kk * (hdot(kk, ref.astype(np.complex128)) / hdot(kk, kk))
    e1 = e1 / norm(e1)
    e2 = np.cross(np.real(kk), np.real(e1)).astype(np.complex128)
    e2 = e2 / norm(e2)
    return e1, e2


def helical_basis(k: Iterable[int] | Array) -> Tuple[Array, Array]:
    e1, e2 = orthonormal_frame_perp(k)
    # convention: h+ = (e1 + i e2)/sqrt(2), h-=(e1 - i e2)/sqrt(2)
    hp = (e1 + 1j * e2) / np.sqrt(2.0)
    hm = (e1 - 1j * e2) / np.sqrt(2.0)
    return hp, hm


def helical_coeffs(k: Iterable[int] | Array, v: Array) -> tuple[complex, complex]:
    hp, hm = helical_basis(k)
    return hdot(hp, v), hdot(hm, v)


def helicity_purity(k: Iterable[int] | Array, v: Array) -> float:
    """Return max helicity-energy fraction in [0,1]."""
    cp, cm = helical_coeffs(k, v)
    ep, em = abs(cp) ** 2, abs(cm) ** 2
    if ep + em <= 1e-15:
        return 0.0
    return max(ep, em) / (ep + em)
