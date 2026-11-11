import numpy as np

from circulation.verify_identity import averaged_circulation, w_heat


def test_heat_vorticity_is_average_circulation():
    x = np.array([0.3, -0.2, 0.1])
    n = np.array([1.0, 2.0, -1.0]) / np.sqrt(6.0)
    t = 0.2
    lhs = 4 * np.pi * t * float(np.dot(np.asarray(w_heat(*x, t), float).ravel(), n))
    rhs = averaged_circulation(x, n, t, nh=50, nr=50)
    assert abs(lhs - rhs) < 1e-10 * abs(lhs)
