# Circulation and the largest critical space for Navier–Stokes

Status: a short note with complete proofs of an elementary identity and two
corollaries of published theorems. It does **not** prove regularity or blowup.
The mathematical content beyond the cited theorems is the identity in §1; the
corollaries in §3 are special cases of known criteria, restated in terms of a
physical quantity. Novelty is unverified (see §5).

Notation: $u$ is a divergence-free velocity field on $\mathbb{R}^3$,
$\omega = \nabla\times u$, $e^{t\Delta}$ is the heat semigroup, and for a
circle $C$ with unit normal $n$, $\Gamma_C(u)=\oint_C u\cdot dl$ is the
circulation, taken right-handed about $n$. $M_R(u)$ is the supremum of
$|\Gamma_C(u)|$ over all circles of radius at most $R$, and
$M_\infty(u)$ the supremum over all circles.

## 1. The identity

**Proposition 1.** Let $u\in C^1$ with $\omega$ bounded and continuous. For
every $x\in\mathbb{R}^3$, unit vector $n$ and $t>0$,

$$4\pi t\,\big(e^{t\Delta}\omega\big)(x)\cdot n \;=\; \mathbb{E}\,\Gamma_{C}(u),$$

where $C$ is a random circle with normal $n$, centre $x+hn$ with
$h\sim N(0,2t)$, and radius $\rho$ with density
$p_t(\rho)=\frac{\rho}{2t}e^{-\rho^2/4t}$ on $(0,\infty)$, independent of $h$.

*Proof.* Write $y=x+hn+z$ with $z\perp n$. The heat kernel factors as
$G_t(y-x)=g_1(h)\,g_2(|z|)$ with $g_1(h)=(4\pi t)^{-1/2}e^{-h^2/4t}$ and
$g_2(s)=(4\pi t)^{-1}e^{-s^2/4t}$. Since $-g_2'(\rho)=\frac{1}{4\pi t}p_t(\rho)$,

$$g_2(s)=\int_s^\infty -g_2'(\rho)\,d\rho=\frac{1}{4\pi t}\int_0^\infty \mathbf{1}_{\{s<\rho\}}\,p_t(\rho)\,d\rho .$$

Insert this and use Fubini ($\omega$ is bounded):

$$\big(e^{t\Delta}\omega\big)(x)\cdot n=\frac{1}{4\pi t}\int g_1(h)\,dh\int_0^\infty p_t(\rho)\,d\rho\int_{|z|<\rho}\omega(x+hn+z)\cdot n\,dz .$$

The inner integral is the flux of $\omega$ through the disc of radius $\rho$
centred at $x+hn$, which equals $\Gamma_C(u)$ by Stokes' theorem. $g_1$ and
$p_t$ are probability densities. $\qquad\blacksquare$

So the heat-smoothed vorticity at a point is $1/(4\pi t)$ times an average of
circulations around coaxial circles of radius about $\sqrt{t}$.

**Numerical check.** `verify_identity.py` compares both sides for a
non-symmetric test field at 8 random $(x,n,t)$; the worst relative difference
is $4.4\times10^{-12}$.

## 2. Consequences of the identity

**Corollary 2 (sharp bound).** $\;4\pi t\,\|e^{t\Delta}\omega\|_\infty\le M_\infty(u)$ for all $t>0$.

*Proof.* $|v|=\max_n v\cdot n$, and an average is at most a supremum. $\blacksquare$

The constant $4\pi$ is optimal: for a straight line vortex of circulation
$\Gamma$, every circle linked once with the line has circulation $\pm\Gamma$,
unlinked circles have none, and $e^{t\Delta}\omega=\Gamma/(4\pi t)$ on the axis.
The line vortex has infinite energy, but thin vortex rings of large radius
have finite energy and approach the same ratio.

**Corollary 3 (localized bound).** If also $u\in L^2$, then for all $R>0$, $t>0$,

$$4\pi t\,\|e^{t\Delta}\omega\|_\infty\le M_R(u)+\|u\|_{L^2}\left(\frac{\pi(1+a)e^{-a}}{\sqrt{8\pi t}}\right)^{1/2},\qquad a=\frac{R^2}{2t}.$$

*Proof.* Split the average at $\rho=R$. For $\rho>R$ use
$|\Gamma_C|\le\oint_C|u|\,dl$. Since $dl\,d\rho$ is the area element in the
plane of the circle, the tail is $\int |u(y)|\,W(y)\,dy$ with
$W=g_1(h)\,p_t(|z|)\mathbf{1}_{\{|z|>R\}}$, and by Cauchy–Schwarz it is at most
$\|u\|_2\|W\|_2$. Directly,
$\|W\|_2^2=(8\pi t)^{-1/2}\cdot\pi\int_a^\infty se^{-s}ds=(8\pi t)^{-1/2}\pi(1+a)e^{-a}$. $\blacksquare$

For fixed $R$ the second term tends to $0$ as $t\to0$.

**Corollary 4 (critical Besov norm).** For divergence-free $u\in L^2$ as above,

$$\sup_{t>0}\sqrt{t}\,\|e^{t\Delta}u\|_\infty\le C\,M_\infty(u),$$

with $C$ absolute. The left side is the standard heat-kernel norm of
$\dot B^{-1}_{\infty,\infty}$, the largest critical space.

*Proof.* Since $-\Delta u=\nabla\times\omega$ and $e^{s\Delta}u\to0$,
$e^{t\Delta}u=\int_t^\infty\nabla\times e^{s\Delta}\omega\,ds$. With
$N=\sup_s s\|e^{s\Delta}\omega\|_\infty$,
$\|\nabla\times e^{s\Delta}\omega\|_\infty\le Cs^{-1/2}\|e^{(s/2)\Delta}\omega\|_\infty\le 2CNs^{-3/2}$,
and $\int_t^\infty s^{-3/2}ds=2t^{-1/2}$. Then apply Corollary 2. $\blacksquare$

**High-frequency form.** Let $u_q$ be the Littlewood–Paley block at frequency
$\lambda_q=2^q$ (as in Cheskidov–Shvydkoy). Then
$u_q=\lambda_q^{-1}m(D/\lambda_q)\,e^{t\Delta}\omega$ with $t=\lambda_q^{-2}$
and $m(\xi)=\varphi(\xi)e^{|\xi|^2}\,i\xi\times/|\xi|^2$, a smooth compactly
supported symbol. Hence

$$\lambda_q^{-1}\|u_q\|_\infty\le C_0\,t\,\|e^{t\Delta}\omega\|_\infty,\qquad t=\lambda_q^{-2},\tag{1}$$

with $C_0=\|\mathcal{F}^{-1}m\|_{L^1}$ absolute.

## 3. Corollaries of published theorems

**Theorem A (Cheskidov–Shvydkoy, Lemma 3.2 of arXiv:0708.3067).** There is an
absolute $c>0$ such that a Leray–Hopf solution on $[0,T]$ with
$\limsup_{q\to\infty}\sup_{t\in(0,T)}\lambda_q^{-1}\|u_q(t)\|_\infty<c\nu$ is
regular on $(0,T]$.

**Corollary 5 (small circulation on small circles prevents blowup).** Let $u$
be a Leray–Hopf solution that is a strong solution on $(0,T)$ (bounded, with
bounded derivatives, on compact time subintervals). Suppose there are $R>0$ and
$\delta>0$ with

$$\sup_{T-\delta<t<T}M_R\big(u(t)\big)<c_1\nu,\qquad c_1=\frac{4\pi c}{C_0}.$$

Then $u$ is regular at $T$.

*Proof.* Apply Theorem A on $[T-\delta,T]$. By (1) and Corollary 3 with
$t=\lambda_q^{-2}$,
$\lambda_q^{-1}\|u_q(s)\|_\infty\le\frac{C_0}{4\pi}\big[M_R(u(s))+\|u_0\|_2\,\tau_q\big]$
where $\tau_q\to0$ as $q\to\infty$ and the energy inequality gives
$\|u(s)\|_2\le\|u_0\|_2$. $\blacksquare$

Read contrapositively: **at a first singular time $T$, for every $R>0$ there
are circles of radius at most $R$ and times approaching $T$ at which
$|\Gamma_C|\ge c_1\nu$.** A singularity needs a circulation Reynolds number
$\Gamma/\nu$ of order one at arbitrarily small scales. For a smooth field the
circulation around a circle of radius $\rho$ is about $\pi\rho^2|\omega|$, so
this is a scale-invariant lower bound on concentration.

**Theorem B (Seregin–Zhou, arXiv:1802.03600, as stated in its abstract).** A
suitable weak solution in $L_\infty(0,T;\dot B^{-1}_{\infty,\infty})$ has all
scaled energy quantities bounded (so any singularity is of Type I), and is
smooth if axially symmetric.

**Corollary 6 (Type II blowup needs unbounded circulation).** If
$\sup_{t<T}M_\infty(u(t))<\infty$, then by Corollary 4 the solution is in
$L_\infty(0,T;\dot B^{-1}_{\infty,\infty})$ and Theorem B applies. So a Type II
singularity requires circles whose circulation is unbounded as $t\to T$.
I have read only the abstract of Seregin–Zhou; this corollary is conditional
on the theorem as stated there.

## 4. Two remarks

**Pressure-free evolution.** For a fixed circle,
$\frac{d}{dt}\Gamma_C=\oint_C\big(u\times\omega-\nu\,\nabla\times\omega\big)\cdot dl$,
because $\partial_tu=u\times\omega-\nabla(p+\tfrac12|u|^2)-\nu\nabla\times\omega$
and gradients integrate to zero. The pressure does not appear.

**Consistency with the self-similar window.** For a collapse with length
$(T-t)^\gamma$ and velocity $(T-t)^{\gamma-1}$, core circulation scales as
$(T-t)^{2\gamma-1}$. Viscosity is negligible only for $\gamma<1/2$, exactly
where this diverges, in agreement with Corollary 6.

## 5. What this is and is not

- Corollary 5 is logically **weaker** than Theorem A: its hypothesis implies
  theirs. Its content is the translation into circulation, which is local,
  pressure-free and directly measurable in simulations.
- Whether bounded (not small) circulation implies regularity is the open
  $L_\infty\dot B^{-1}_{\infty,\infty}$ problem in another form. Nothing here
  bears on it.
- The constants $c$, $C_0$, $c_1$ are absolute but not computed.
- Prior art: two web searches for circulation-based criteria in critical
  spaces found none. Closest results found: Cheskidov–Shvydkoy (2010), Zhang
  (arXiv:1210.2790, smallness in the largest class), Seregin–Zhou (2018),
  Farhat–Grujić–Leitmeyer (volumetric sparseness and $B^{-1}_{\infty,\infty}$).
  From memory, not checked: Giga–Miyakawa (vortex filaments of small
  circulation), Migdal (loop equations). An identity this simple may well be
  folklore.
