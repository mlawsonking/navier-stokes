"""Does the universal cusp inner problem admit ASYMMETRIC solutions?  (test for the end point of the CCF family)
    v'(xi) = H(1/v),  v(0) = 1,  v'(0) = 0 (fixes the translation),  v ~ c_+- sqrt(2|xi|) as xi -> +-inf,  c_+ c_- = 1.
The outer cusp solutions (gpu_solver/ccf_cusp.py) come in a one-parameter family labelled by A_R/A_L = c_-/c_+;
if the inner problem only has the even solution, only A_R = A_L can be the end of the regular family.
Method: as runs/cusp_inner.py but on the full line. w = 1/v = w0 + w1, w0 = Re[C (xi + i b)^(-1/2)] with
Re C = 1/(sqrt2 c_+), Im C = 1/(sqrt2 c_-) (H w0 = Im[...] exactly); w1 unknown at +-xi_j (xi_j = exp(tau_j));
H w1 by the half-step trapezoid rule on its even and odd parts; v by the midpoint rule from xi = 0; rows
w v - 1 = 0 at all nodes plus v'(0) = 0; Gauss-Newton (least squares, N+1 rows, N unknowns).
Usage: cusp_inner_asym.py RATIO [h]     RATIO = c_+/c_-  (1 = the even solution of cusp_inner.py)"""
import sys, math, torch
torch.set_default_dtype(torch.float64)
import os; WT = float(os.environ.get("WT", 10.0))   # weight of the v'(0) = 0 row (0: translation left free)
ratio = float(sys.argv[1]); h = float(sys.argv[2]) if len(sys.argv) > 2 else 0.04; b = 1.0
cp, cm = math.sqrt(ratio), 1 / math.sqrt(ratio); Cc = complex(1 / (math.sqrt(2) * cp), 1 / (math.sqrt(2) * cm))
tau = torch.arange(math.log(1e-7), math.log(1e9), h); ys = torch.exp(tau); yc = torch.exp(tau[:-1] + h / 2); n = ys.numel()
def pair(x):                                     # w0 and H w0 at real x (tensor), from C (x + i b)^(-1/2)
    z = torch.complex(x, torch.full_like(x, b)); f = Cc * z ** (-0.5); return f.real, f.imag
Ke = (2 * yc[:, None] / math.pi) * (h * ys)[None, :] / (yc[:, None] ** 2 - ys[None, :] ** 2)
Ko = (2 / math.pi) * (h * ys * ys)[None, :] / (yc[:, None] ** 2 - ys[None, :] ** 2)
w0p, _ = pair(ys); w0m, _ = pair(-ys); _, H0p = pair(yc); _, H0m = pair(-yc); _, H00 = pair(torch.zeros(1))
def parts(w1):
    wp, wm = w1[:n], w1[n:]; e, o = (wp + wm) / 2, (wp - wm) / 2
    Hp = H0p + Ke @ e + Ko @ o; Hm = H0m - Ke @ e + Ko @ o             # v' at +yc and -yc
    vp = 1 + torch.cat([torch.zeros(1), torch.cumsum(Hp * h * yc, 0)])
    vm = 1 - torch.cat([torch.zeros(1), torch.cumsum(Hm * h * yc, 0)])
    vp0 = H00 - (2 / math.pi) * (h * o).sum()                          # v'(0) = (H w)(0); H(odd)(0) = -(2/pi) int o/t
    return vp, vm, vp0
def res(w1):
    vp, vm, vp0 = parts(w1)
    return torch.cat([(w0p + w1[:n]) * vp - 1, (w0m + w1[n:]) * vm - 1, WT * vp0])
x0 = 1 / torch.sqrt(1 + 2 * ys)
w1 = torch.cat([x0 / cp - w0p, x0 / cm - w0m]) if False else torch.cat([1 / torch.sqrt(1 + 2 * ys) - w0p, 1 / torch.sqrt(1 + 2 * ys) - w0m])
for it in range(60):
    r = res(w1); Jm = torch.autograd.functional.jacobian(res, w1)
    st = torch.linalg.lstsq(Jm, -r[:, None]).solution[:, 0]; t = 1.0
    while t > 1e-4 and not res(w1 + t * st).norm() < r.norm(): t /= 2
    w1 = w1 + t * st; rn = res(w1).abs().max().item()
    if it % 5 == 0 or rn < 1e-12: print(f"  it {it} sup {rn:.2e} t {t}", flush=True)
    if rn < 1e-12 or t <= 1e-4: break
vp, vm, vp0 = parts(w1)
print(f"ratio c+/c- = {ratio}: newton its {it + 1} final sup {rn:.2e}  v'(0) {vp0.item():.1e}")
for x in (1e2, 1e4, 1e6, 1e8):
    i = (ys - x).abs().argmin().item(); s2 = math.sqrt(2 * ys[i].item())
    print(f"   xi {ys[i].item():9.2e}  v(+xi)/sqrt(2xi) {vp[i].item()/s2:.5f} (c+ {cp:.5f})  v(-xi)/sqrt(2xi) {vm[i].item()/s2:.5f} (c- {cm:.5f})")
for lev in (2.0, 3.0):
    ip = (vp > lev).nonzero()[0].item(); im = (vm > lev).nonzero()[0].item()
    print(f"   v = {lev}: xi+ {ys[ip].item():.4f}  xi- {-ys[im].item():.4f}")
rr = res(w1)[:2 * n].abs(); rp, rm = rr[:n], rr[n:]
print("   |residual| by decade of xi (+side / -side):")
for lo in range(-7, 9):
    m = (ys >= 10.0 ** lo) & (ys < 10.0 ** (lo + 1))
    if m.any(): print(f"     1e{lo:+d}: {rp[m].max().item():.1e} / {rm[m].max().item():.1e}")
