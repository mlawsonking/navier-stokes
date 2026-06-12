"""Exact finite-dimensional Fourier-symbol utilities for incompressible Navier--Stokes.

The conventions here intentionally omit harmless global constants such as -i.
We focus on geometry: Leray projection, output directions, relative sizes,
path collisions, and polarization/vorticity defects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

Array = np.ndarray
Mode = Tuple[int, int, int]


def vec(x: Iterable[float | complex]) -> Array:
    """Return a 3-vector as a numpy array."""
    return np.asarray(tuple(x), dtype=np.complex128)


def real_vec(x: Iterable[float]) -> Array:
    """Return a real 3-vector as a complex vector for uniform operations."""
    return np.asarray(tuple(x), dtype=np.float64).astype(np.complex128)


def mode(k: Iterable[int]) -> Array:
    return np.asarray(tuple(k), dtype=np.float64).astype(np.complex128)


def norm2(v: Array) -> float:
    return float(np.real(np.vdot(v, v)))


def norm(v: Array) -> float:
    return float(np.sqrt(max(norm2(v), 0.0)))


def dot(a: Array, b: Array) -> complex:
    """Bilinear dot product, not Hermitian. Correct for Fourier symbol algebra."""
    return complex(np.dot(a, b))


def hdot(a: Array, b: Array) -> complex:
    """Hermitian inner product for metric/norm calculations."""
    return complex(np.vdot(a, b))


def leray_project(k: Iterable[int] | Array, v: Array, *, tol: float = 1e-12) -> Array:
    """Project v onto k^perp using the Euclidean/Hermitian metric."""
    kk = mode(k) if not isinstance(k, np.ndarray) else k.astype(np.complex128)
    denom = hdot(kk, kk)
    if abs(denom) < tol:
        return v.copy()
    return v - kk * (hdot(kk, v) / denom)


def is_div_free(k: Iterable[int] | Array, a: Array, *, tol: float = 1e-9) -> bool:
    kk = mode(k) if not isinstance(k, np.ndarray) else k.astype(np.complex128)
    return abs(dot(kk, a)) < tol


def ns_bilinear(p: Iterable[int] | Array, a: Array, q: Iterable[int] | Array, b: Array) -> Array:
    """Projected NS bilinear output at p+q.

    B(p,a;q,b) = P_{p+q}[(a.q)b + (b.p)a].
    This is the symmetrized contribution of the two input modes.
    """
    pp = mode(p) if not isinstance(p, np.ndarray) else p.astype(np.complex128)
    qq = mode(q) if not isinstance(q, np.ndarray) else q.astype(np.complex128)
    raw = dot(a, qq) * b + dot(b, pp) * a
    return leray_project(pp + qq, raw)


def cross(a: Array, b: Array) -> Array:
    return np.cross(a, b)


def vorticity_polarization(k: Iterable[int] | Array, a: Array) -> Array:
    """Vorticity polarization, ignoring the global factor i: k x a."""
    kk = mode(k) if not isinstance(k, np.ndarray) else k.astype(np.complex128)
    return cross(kk, a)


def angle_between(a: Array, b: Array, *, projective: bool = True, tol: float = 1e-12) -> float:
    """Return angle in radians between complex vectors using Hermitian metric.

    If projective=True, phase/sign is ignored, so the result lies in [0, pi/2].
    For real vectors this is the acute line angle.
    """
    na, nb = norm(a), norm(b)
    if na < tol or nb < tol:
        return float("nan")
    c = abs(hdot(a, b)) / (na * nb) if projective else np.real(hdot(a, b)) / (na * nb)
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.arccos(c))


def one_line_miss_fraction(vectors: Iterable[Array], weights: Iterable[float] | None = None) -> float:
    """Minimum squared energy fraction missed by any one complex line.

    For vectors in a 2D plane, this is lambda_min / trace of the covariance.
    Works in C^3 with Hermitian covariance; it reports all energy not captured
    by the best rank-one line.
    """
    vecs = [np.asarray(v, dtype=np.complex128) for v in vectors]
    if weights is None:
        weights = [1.0] * len(vecs)
    cov = np.zeros((3, 3), dtype=np.complex128)
    total = 0.0
    for v, w in zip(vecs, weights):
        ww = float(w)
        cov += ww * np.outer(v, np.conjugate(v))
        total += ww * norm2(v)
    if total <= 1e-14:
        return 0.0
    vals = np.linalg.eigvalsh(cov)
    # best line captures largest eigenvalue
    missed = max(total - float(vals[-1]), 0.0)
    return missed / total


def line_capture_fraction(vectors: Iterable[Array], weights: Iterable[float] | None = None) -> float:
    return 1.0 - one_line_miss_fraction(vectors, weights)


def squared_radius(k: Iterable[int] | Array) -> int:
    kk = np.asarray(tuple(k), dtype=int) if not isinstance(k, np.ndarray) else np.rint(np.real(k)).astype(int)
    return int(np.dot(kk, kk))


@dataclass(frozen=True)
class Interaction:
    p: Mode
    q: Mode
    m: Mode
    output: Array

    @property
    def output_norm(self) -> float:
        return norm(self.output)

    @property
    def radius2(self) -> int:
        return squared_radius(self.m)
