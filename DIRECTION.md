# Direction for the GPU solver work

Purpose: say which computations are worth running, what each must reproduce
before its output is trusted, and when to stop. Written for the builder
session on the RTX 4080. Nothing here is a result.

Conventions. A self-similar collapse is written

    u(x,t) = (T-t)^lambda * U(y),   y = x / (T-t)^(1+lambda)

so the length scale is (T-t)^gamma with gamma = 1 + lambda. Reference values
marked "exact" were derived by hand in this file's parent session and can be
checked on paper. Values marked "published" must be read from the cited paper
by the builder; I have not verified the digits.

## Rung 1: exactly solvable cases (must pass before anything else)

1a. Burgers, u_t + u u_x = 0.
    Profile equation: -lambda*U + ((1+lambda)*y + U) * U' = 0.
    Exact: smooth solutions exist only for lambda = 1/(2i), i = 1,2,3,...,
    given implicitly by y = -U - U^(1+1/lambda) (normalised U'(0) = -1).
    Stable case lambda = 1/2: y = -U - U^3.
    Pass: with lambda free and only smoothness imposed, the solver returns
    lambda = 0.5 to 1e-6 and matches the implicit profile to 1e-6 in sup norm
    on |y| <= 20. Then, imposing U'''(0) = 0, it returns lambda = 0.25.
    Why: tests that the solver finds the exponent as an unknown, including an
    unstable branch.

1b. Constantin-Lax-Majda, w_t = w * H(w), H the Hilbert transform with kernel
    p.v. 1/(pi*(x-y)).
    Profile equation: W + c*y*W' = W * H(W), with w = W(y)/(T-t), y = x/(T-t)^c.
    Exact, c = 1: W(y) = -2y/(1+y^2), H(W) = 2/(1+y^2).
    Pass: residual below 1e-8 and profile error below 1e-6 on |y| <= 50.
    Why: tests the nonlocal operator on an unbounded domain. Every later
    target has a nonlocal velocity law, so a solver that fails here is not
    ready.

## Rung 2: reproduce published profiles

2a. Cordoba-Cordoba-Fontelos (1D, nonlocal). Published: stable and first
    unstable exponents in arXiv:2509.14185.
2b. 2D Boussinesq with boundary (equivalently 3D axisymmetric Euler with a
    cylindrical wall). Published: stable lambda in arXiv:2201.06780; the later
    paper arXiv:2509.14185 fits lambda_n ~ 1 + 1/(1.4187 n + 1.0863), which
    gives about 1.9206 for n = 0.
    Pass for 2a and 2b: published lambda reproduced to at least 4 digits,
    stable under doubling the collocation points and under a second random
    seed.
    Note: gamma = 1 + lambda is about 2.92 here, far above 1/2. Viscosity wins
    against this collapse. It is a solver benchmark, not a Navier-Stokes
    candidate.

## Rung 3: the search that could matter

Target: a smooth self-similar Euler profile in the whole space (no wall),
axisymmetric with swirl, odd in z.

What counts as new:
- any converged smooth profile at all (to my knowledge none is known without
  a boundary for smooth data; unverified);
- of interest for Navier-Stokes only if 2/5 <= gamma < 1/2, that is
  -0.6 <= lambda < -0.5. Below 1/2 viscosity is asymptotically negligible;
  at or above 2/5 the core energy stays finite.

Required checks before calling anything a candidate:
- sup-norm residual below 1e-8, lambda stable to 6 digits across two seeds
  and a doubling of collocation points;
- smooth at the origin (no hidden singular behaviour: check derivatives up to
  order 4);
- far-field decay |U| ~ |y|^(lambda/(1+lambda)), which in the window means a
  decay exponent between -1.5 and -1;
- core circulation: circulation around circles of radius R in profile
  variables, reported as a function of R. In the window the physical core
  circulation must grow like (T-t)^(2*gamma-1); see circulation/README.md on
  branch circulation-note.

Stop conditions:
- Do not start rung 3 until 1a, 1b and 2b pass.
- Budget: 40 GPU-hours and 30 distinct initialisations. If nothing reaches a
  residual of 1e-6 by then, record a null result with the residual history
  and stop.
- A null result is the expected outcome. Report it plainly.

## Reporting

For every run, HANDOFF.md should record: equation and ansatz, lambda found,
final residual (sup norm), points and seeds used, wall time, and whether the
pass condition was met. Record failures as well as successes.
