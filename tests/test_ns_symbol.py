import numpy as np

from navier_stokes_cascade.ns_symbol import leray_project, ns_bilinear, real_vec, norm


def test_leray_projection_perpendicular():
    k = (1, 2, 3)
    v = real_vec((4, 5, 6))
    p = leray_project(k, v)
    assert abs(np.dot(np.array(k, dtype=float), p)) < 1e-9


def test_plane_wave_self_interaction_vanishes():
    k = (1, 0, 0)
    a = real_vec((0, 1, 0))
    out = ns_bilinear(k, a, k, a)
    assert norm(out) < 1e-9
