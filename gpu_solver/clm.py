"""Rung 1b: Constantin-Lax-Majda w_t = w H(w), self-similar w = W(y)/(T-t), y = x/(T-t)^c.

Profile equation  W + c*y*W' = W * H(W).  Exact (c=1): W = -2y/(1+y^2), H W = 2/(1+y^2).

Discretisation: y = tan(theta/2). Then y*d/dy = sin(theta) d/dtheta, and W odd,
decaying  <=>  F(theta) = W(tan(theta/2)) = sum_{k=1}^{K} b_k sin(k theta).
Closed forms used (no FFT needed for a sine series):
    y W'  = sin(th) * sum k b_k cos(k th)
    H W   = -sum b_k cos(k th) + sum b_k (-1)^k        (Cayley map, see hilbert.py)
Normalisation (y-scaling symmetry): W'(0) = -2  <=>  sum k b_k = -1.
Unknowns: b_1..b_K and optionally c.
"""
import math
import torch

DT = torch.float64


def colloc_theta(M, device="cuda"):
    j = torch.arange(M, dtype=DT, device=device)
    return math.pi * (j + 0.5) / M  # (0, pi); oddness covers (-pi, 0)


def fields(b, th):
    K = b.shape[0]
    k = torch.arange(1, K + 1, dtype=DT, device=b.device)
    S = torch.sin(th[:, None] * k)
    C = torch.cos(th[:, None] * k)
    F = S @ b
    yWy = torch.sin(th) * (C @ (k * b))
    HW = -(C @ b) + (b * (-1.0) ** k).sum()
    return F, yWy, HW


def residual_vec(params, th, c_fixed=None, w_norm=1.0):
    if c_fixed is None:
        c, b = params[0], params[1:]
    else:
        c, b = torch.as_tensor(c_fixed, dtype=DT, device=params.device), params
    F, yWy, HW = fields(b, th)
    R = F + c * yWy - F * HW
    K = b.shape[0]
    k = torch.arange(1, K + 1, dtype=DT, device=b.device)
    norm = (w_norm * ((k * b).sum() + 1.0)).reshape(1)
    return torch.cat([R, norm])
