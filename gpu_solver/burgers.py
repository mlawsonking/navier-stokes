"""Rung 1a: self-similar profiles of inviscid Burgers, u_t + u u_x = 0.

Ansatz u = (T-t)^lam U(y), y = x/(T-t)^(1+lam). Profile equation
    R(U) = -lam*U + ((1+lam)*y + U) * U' = 0.
Exact smooth sublinear solutions: lam = 1/(2i), y = -U - C*U^(1+1/lam).

Representation (U odd, smooth at 0, sublinear at infinity):
    a = 1/(1+lam),  s = (1+y^2)^(-a/2)  in (0,1]   (s=1 at y=0, s ~ y^-a at inf)
    U = -y * s * N(s),   N(s) = 1 + s(1-s) * sum_k c_k T_k(2s-1)
N(1)=1 fixes U'(0)=-1; N(0)=1 fixes the far field U ~ -y^(lam/(1+lam)), i.e. C=1.
Both are normalisations of the two symmetries (y-scaling is the only continuous
one; the trivial solution U=-y is excluded because it is not sublinear).
lam is an unknown. Analyticity in s at s=1 is the smoothness condition at y=0.

Optional constraint U'''(0)=0  <=>  sum_k c_k = 1 (derivation in HANDOFF.md).
"""
import math
import torch

DT = torch.float64


def cheb_T_dT(x, n):
    """T_k(x), T_k'(x) for k<n on x in (-1,1)."""
    th = torch.arccos(x)
    k = torch.arange(n, dtype=x.dtype, device=x.device)
    T = torch.cos(th[:, None] * k)
    dT = k * torch.sin(th[:, None] * k) / torch.sin(th)[:, None]
    return T, dT


def profile(params, y, n):
    lam, c = params[0], params[1:]
    a = 1.0 / (1.0 + lam)
    s = (1.0 + y * y) ** (-a / 2)
    T, dT = cheb_T_dT(2 * s - 1, n)
    P = T @ c
    Ps = 2 * (dT @ c)
    q = s - s * s
    N = 1 + q * P
    Ns = (1 - 2 * s) * P + q * Ps
    U = -y * s * N
    Uy = -s * N + y * y * s * (N + s * Ns) * a / (1 + y * y)
    return U, Uy, s


def residual_vec(params, y, n, u3_zero=False, w_con=1.0):
    lam = params[0]
    U, Uy, s = profile(params, y, n)
    R = (-lam * U + ((1 + lam) * y + U) * Uy) / (y * s)
    if u3_zero:
        R = torch.cat([R, (w_con * (params[1:].sum() - 1.0)).reshape(1)])
    return R


def collocation_y(n_pts, lam_ref=0.5, device="cuda"):
    """Chebyshev points in s on (0,1), mapped to y>0 with the reference lam."""
    j = torch.arange(n_pts, dtype=DT, device=device)
    s = 0.5 * (1 - torch.cos(math.pi * (j + 0.5) / n_pts))
    a = 1.0 / (1.0 + lam_ref)
    return torch.sqrt(s ** (-2.0 / a) - 1.0)


def exact_U(y, m):
    """Real root of U + U^m + y = 0 (m odd) by vectorised bisection; numpy-free."""
    lo = -y.abs() - 1.0
    hi = y.abs() + 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f = mid + mid ** m + y
        pos = f > 0
        hi = torch.where(pos, mid, hi)
        lo = torch.where(pos, lo, mid)
    return 0.5 * (lo + hi)


def residual_vec_smooth(params, y, n, u3_zero=False, y_near=None, kmax=6, w_der=1.0):
    """residual_vec plus d^k R / dy^k (k=1..kmax) at points y_near.

    Why: for lam not of the form 1/(2i) the exact sublinear solutions contain
    |U|^(1+1/lam), which is C^m at y=0 with m = floor(1/lam). A large polynomial
    basis can imitate it with small residual R, but the imitation has large
    high-order derivatives of R near 0. Exact smooth solutions have R^(k) = 0.
    """
    from torch.func import jacfwd, vmap
    base = residual_vec(params, y, n, u3_zero)
    if y_near is None:
        return base

    def R1(yy):  # unnormalised residual at a single point
        yv = yy.reshape(1)
        U, Uy, s = profile(params, yv, n)
        return (-params[0] * U + ((1 + params[0]) * yv + U) * Uy)[0]

    rows = []
    f = R1
    for k in range(1, kmax + 1):
        f = jacfwd(f)
        rows.append(w_der * vmap(f)(y_near) / math.factorial(k))
    return torch.cat([base] + rows)


def cheb_T_dT_rec(x, n):
    """T_k, T_k' by three-term recurrence; valid for complex x (used off the real axis)."""
    T = [torch.ones_like(x), x]
    dT = [torch.zeros_like(x), torch.ones_like(x)]
    for k in range(1, n - 1):
        T.append(2 * x * T[k] - T[k - 1])
        dT.append(2 * T[k] + 2 * x * dT[k] - dT[k - 1])
    return torch.stack(T[:n], -1), torch.stack(dT[:n], -1)


def residual_complex(params, z, n):
    """Unnormalised profile residual at complex points z (|Im z| < 1, Re z > 0).

    The exact smooth profile is analytic in a strip around the real axis, so its
    residual vanishes there too. A real-axis imitation of a non-smooth solution
    (|U|^(1+1/lam) at y=0) does not continue analytically and shows up here.
    Returned as [Re, Im] stacked, normalised by |z s|.
    """
    lam, c = params[0], params[1:].to(z.dtype)
    a = 1.0 / (1.0 + lam)
    s = (1.0 + z * z) ** (-a / 2)
    T, dT = cheb_T_dT_rec(2 * s - 1, n)
    P = T @ c
    Ps = 2 * (dT @ c)
    q = s - s * s
    N = 1 + q * P
    Ns = (1 - 2 * s) * P + q * Ps
    U = -z * s * N
    Uy = -s * N + z * z * s * (N + s * Ns) * a / (1 + z * z)
    R = (-lam * U + ((1 + lam) * z + U) * Uy) / (z * s).abs()
    return torch.cat([R.real, R.imag])


def residual_vec_analytic(params, y, n, u3_zero=False, z=None, w=1.0):
    base = residual_vec(params, y, n, u3_zero)
    if z is None:
        return base
    return torch.cat([base, w * residual_complex(params, z, n)])


# ---- origin Taylor rows (math session, HANDOFF Q1) -------------------------------
def _smul(a, b):
    """Truncated product of two power series (coefficient vectors of equal length)."""
    M = a.shape[0]
    return torch.stack([sum(a[i] * b[j - i] for i in range(j + 1)) for j in range(M)])


def cheb_origin_table(n, M, device="cuda"):
    """d[k,j]: T_k(1+2x) = sum_j d[k,j] x^j,  d[k,j] = prod_{i<j} (k^2-i^2)/((i+1/2)(i+1))."""
    k = torch.arange(n, dtype=DT, device=device)
    cols = [torch.ones(n, dtype=DT, device=device)]
    for i in range(M - 1):
        cols.append(cols[-1] * (k * k - i * i) / ((i + 0.5) * (i + 1)))
    return torch.stack(cols, 1)


def origin_series(params, D):
    """Taylor coefficients v_j of V(t), U = y V(y^2), t = y^2, from the ansatz, exactly
    (power-series arithmetic, no autodiff). u_(2j+1) = v_j; v_0 = -1."""
    lam, c = params[0], params[1:]
    M = D.shape[1]
    a = 1.0 / (1.0 + lam)
    b = [torch.ones_like(lam)]
    for j in range(1, M):
        b.append(b[-1] * (-a / 2 - j + 1) / j)
    s = torch.stack(b)                                   # s(t) = (1+t)^(-a/2)
    e0 = torch.zeros(M, dtype=s.dtype, device=s.device); e0[0] = 1.0
    dl = s - e0                                          # s - 1 = O(t)
    pj = c @ D                                           # P = sum_j pj (s-1)^j
    Pt, pw = pj[0] * e0, e0
    for j in range(1, M):
        pw = _smul(pw, dl)
        Pt = Pt + pj[j] * pw
    N = e0 - _smul(_smul(s, dl), Pt)                     # N = 1 + s(1-s) P
    return -_smul(s, N)


def origin_rows(params, D):
    """Coefficients of t^1..t^(M-1) in R/y = -lam V + ((1+lam) + V)(V + 2 t V'):
    r3, r5, r7, ...  Also returns v (so u3 = v[1])."""
    lam = params[0]
    v = origin_series(params, D)
    M = v.shape[0]
    j = torch.arange(M, dtype=v.dtype, device=v.device)
    Up = (1 + 2 * j) * v                                 # series of U'
    e0 = torch.zeros(M, dtype=v.dtype, device=v.device); e0[0] = 1.0
    Rs = -lam * v + _smul((1 + lam) * e0 + v, Up)
    return Rs[1:], v


def residual_vec_origin(params, y, n, D, u3_zero=False, w=1.0):
    """Collocation residual + origin Taylor rows at order-one weight. With u3_zero the
    constraint row is u3 = 0 taken from the series."""
    lam = params[0]
    U, Uy, s = profile(params, y, n)
    R = (-lam * U + ((1 + lam) * y + U) * Uy) / (y * s)
    rows, v = origin_rows(params, D)
    parts = [R, w * rows]
    if u3_zero:
        parts.append(w * v[1].reshape(1))
    return torch.cat(parts)
