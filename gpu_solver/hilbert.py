"""Hilbert transform on R, kernel p.v. 1/(pi (x-y)), via the Cayley map.

y = tan(theta/2) sends R to the circle. For f -> 0 at infinity, with
F(theta) = f(tan(theta/2)):
    (H f)(tan(theta/2)) = (H_T F)(theta) - (H_T F)(pi),
H_T the periodic conjugate operator, H_T e^{ik theta} = -i sgn(k) e^{ik theta}.
Checked against closed-form pairs in tests/test_gpu_solver.py.
"""
import torch


def hilbert_R_from_samples(F):
    """F sampled at theta_j = -pi + 2*pi*j/N (j=0..N-1); returns H f at same points."""
    N = F.shape[-1]
    Fh = torch.fft.fft(F)
    k = torch.fft.fftfreq(N, d=1.0 / N).to(F.device)
    mult = -1j * torch.sign(k)
    if N % 2 == 0:
        mult[N // 2] = 0
    HF = torch.fft.ifft(Fh * mult).real
    # value at theta = pi equals value at theta = -pi (index 0) by periodicity
    return HF - HF[..., 0:1]
