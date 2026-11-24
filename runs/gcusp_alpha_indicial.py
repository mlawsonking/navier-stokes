"""Inner problem of the cusp for the alpha-deformed CCF equation theta_t - (Lambda^(alpha-1) H theta) theta_x = 0.
Burgers form survives: -c Theta + ((1+lam) y - U) Theta' = 0, U = H_alpha Theta, H_alpha = Lambda^(alpha-1) H,
c = (1+lam) alpha - 1 (theta = (T-t)^c Theta(x/(T-t)^(1+lam))).  Inner problem at V = 0:  v' = H_alpha(1/v),
v ~ C xi^s0, s0 = 1 - alpha/2 (Omega ~ |d|^-(1-alpha/2)).  H_alpha(|x|^-g) = T(g) sgn(x) |x|^(1-alpha-g),
   T(g) = tan(pi (g+alpha-1)/2) 2^(alpha-1) Gamma((g+alpha-1)/2) Gamma((1-g)/2) / (Gamma(g/2) Gamma((2-g-alpha)/2))
(Riesz potential of |x|^-g, then H(|x|^-g') = tan(pi g'/2) sgn |x|^-g').  C^2 = T(s0)/s0.
Linearising v = C xi^s0 (1 + e xi^-s):   (s0 - s) T(s0) + s0 T(s0 + s) = 0.
alpha = 1: 2s - 1 = tan(pi/4 + pi s/2), s = 1/2 + i w, w = 0.649424.   Usage: gcusp_alpha_indicial.py [alpha ...]"""
import sys, mpmath as mp
mp.mp.dps = 30


def T(g, al):
    return mp.tan(mp.pi * (g + al - 1) / 2) * 2 ** (al - 1) * mp.gamma((g + al - 1) / 2) * mp.gamma((1 - g) / 2) / (mp.gamma(g / 2) * mp.gamma((2 - g - al) / 2))


for al in [mp.mpf(v) for v in (sys.argv[1:] or ["1", "0.9", "0.95", "1.05", "1.1", "0.8", "1.2"])]:
    s0 = 1 - al / 2
    if abs(al - 1) < 1e-12: al = al + mp.mpf("1e-20")         # removable singularity of the formula at alpha = 1
    f = lambda s: (s0 - s) * T(s0, al) + s0 * T(s0 + s, al)
    r = mp.findroot(f, mp.mpc(s0, 0.65))
    C2 = T(s0, al) / s0
    print(f"alpha {float(al):.4f}  s0 {float(s0):.4f}  C^2 {float(C2):.6f}  root s = {complex(r)}  (Re - s0 = {float(r.real - s0):+.2e})  w {float(r.imag):.6f}"
          f"  log-freq in ln(peak) 2w/(2-alpha) = {float(2 * r.imag / (2 - al)):.6f}, turning-point ratio {float(mp.exp(mp.pi * (2 - al) / (2 * r.imag))):.3f}")
