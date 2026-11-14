"""Checks for gpu_solver. Run: .venv/Scripts/python tests/test_gpu_solver.py (pytest also works)."""
import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from gpu_solver.hilbert import hilbert_R_from_samples
from gpu_solver import burgers as B, clm as C

DT = torch.float64
dev = "cuda" if torch.cuda.is_available() else "cpu"


def _grid(N):
    th = -math.pi + 2 * math.pi * torch.arange(N, dtype=DT, device=dev) / N
    return th, torch.tan(th / 2)


def test_hilbert_pairs():
    """Closed-form pairs, kernel p.v. 1/(pi (x-y)): f + i Hf is the boundary value of an
    upper-half-plane analytic function (i/(z+i) and -1/(z+i)^2)."""
    th, y = _grid(256)
    m = th.abs() < math.pi - 1e-12  # skip theta=-pi (y=inf)
    pairs = [(-2 * y / (1 + y * y), 2 / (1 + y * y)),
             (1 / (1 + y * y), y / (1 + y * y)),
             ((1 - y * y) / (1 + y * y) ** 2, 2 * y / (1 + y * y) ** 2)]
    for f, Hf in pairs:
        f = torch.where(m, f, torch.zeros_like(f))
        err = (hilbert_R_from_samples(f) - Hf)[m].abs().max().item()
        assert err < 1e-12, err


def test_clm_closed_form_matches_fft():
    torch.manual_seed(0)
    K = 12
    b = torch.randn(K, dtype=DT, device=dev) / torch.arange(1, K + 1, device=dev) ** 2
    th, y = _grid(128)
    F, _, HW = C.fields(b, th)
    m = th.abs() < math.pi - 1e-12
    err = (hilbert_R_from_samples(F) - HW)[m].abs().max().item()
    assert err < 1e-12, err


def test_clm_exact_profile_zero_residual():
    b = torch.zeros(8, dtype=DT, device=dev); b[0] = -1
    th = C.colloc_theta(32, device=dev)
    assert C.residual_vec(b, th, c_fixed=1.0).abs().max().item() < 1e-14


def test_burgers_exact_profiles_zero_residual():
    """Exact implicit profiles y=-U-U^m satisfy the profile equation (independent of the basis)."""
    y = torch.linspace(1e-3, 50, 2001, dtype=DT, device=dev)
    for lam, m in ((0.5, 3), (0.25, 5)):
        U = B.exact_U(y, m)
        Uy = -1 / (1 + m * U ** (m - 1))
        R = -lam * U + ((1 + lam) * y + U) * Uy
        assert R.abs().max().item() < 1e-12


def test_origin_series_matches_hand_formulas():
    """Origin Taylor rows from series arithmetic equal r3=u3(2l-1), r5=u5(4l-1)+3u3^2, r7=u7(6l-1)+8u3u5."""
    import torch
    from gpu_solver import burgers as B
    torch.manual_seed(0)
    n = 12
    p = torch.cat([torch.tensor([0.37], dtype=torch.float64, device="cuda"), 0.3 * torch.randn(n, dtype=torch.float64, device="cuda")])
    rows, v = B.origin_rows(p, B.cheb_origin_table(n, 4, "cuda"))
    lam, u3, u5, u7 = p[0], v[1], v[2], v[3]
    hand = torch.stack([u3 * (2 * lam - 1), u5 * (4 * lam - 1) + 3 * u3 ** 2, u7 * (6 * lam - 1) + 8 * u3 * u5])
    assert (rows - hand).abs().max() < 1e-10 * hand.abs().max()
    assert abs(v[0] + 1) < 1e-14
    assert abs(u3 - (1 - p[1:].sum()) / (2 * (1 + lam))) < 1e-13


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
