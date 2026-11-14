# Prior-art check: gCLM self-similar exponent vs a, stable and unstable smooth branches

Date of check: 2026-11-14. Method: fetched arXiv abstract/HTML/PDF text (pdftotext on downloaded PDFs; HTML stripped to text) and grepped it. CONFIRMED = seen in fetched source text. UNCONFIRMED = search snippet / memory / not opened.

Convention map. Ours: omega=(T-t)^-1 Omega(x/(T-t)^(1+lambda)), so c_l = alpha = 1+lambda.
- Lushnikov-Silantyev-Siegel (LSS) "gamma", Huang-Qin-Wang-Wei (HQWW) "c_l", Huang-Tong-Wang "c_l/gamma": all equal our 1+lambda. CONFIRMED indirectly: LSS Table 1 gives gamma=1 at a=0, 1/3 at a=1/2, 0 at a_c; HQWW state c_l=1, 1/3, -1 for a=0, 1/2, 1; the 2610.30492 ansatz (their eq. 11) is literally ours with lambda. The LSS defining equation (6) was lost in pdftotext (Greek glyphs), so the identification rests on these matches.
- Zheng / Elgindi-Jeong: scaling gamma = 1/alpha, i.e. c_l = 1/alpha. Different "alpha" from the one in the task text.

## ANSWER to (ii): unstable/higher smooth branches for a != -1

None found. No paper I could open gives values, curves, or even an existence statement for a higher/"excited" smooth self-similar branch at any a other than -1. Details:
- 2201.06780 (Wang et al.): unstable solutions only at a=-1 (lambda=0.60573). For a in [-1,1] it only reproduces the LSS stable curve. CONFIRMED.
- 2509.14185, 2511.22819, 2610.30492: unstable CCF branches only at a=-1; none varies a. CONFIRMED (2610.30492 treats only a=0 and a=-1, at prescribed lambda).
- 2603.25104 (Huang-Tong-Wang) says explicitly that the DeepMind work "numerically discovered a stable self-similar profile plus multiple unstable self-similar profiles for [CCF] a=-1" and that uniqueness of the stable one is not established. CONFIRMED. It treats no a-dependence of unstable smooth branches.
- Only a-dependent "unstable" in the literature is a different object: 2603.25104 finds, for degenerate data (vanishing order k=3) with a>0, critical values a_c1 ~ 0.233 and a_c2 ~ 0.751, and a "flipped" compactly supported profile that is unstable for a>a_c2 (CONFIRMED text). It is a sign-flipped/degenerate-data profile, not the CCF excited branch; do not confuse.
- Sign-changing / infinitely many profiles exist at a=1 only (Huang-Tong-Wei, arXiv 2209.08232 abstract CONFIRMED: infinitely many compactly supported self-similar De Gregorio solutions; eigenfunctions of a compact self-adjoint operator). HQWW say sign-changing profiles for a<1 are not excluded but not found (CONFIRMED).

So your lambda_1(a) for -1<=a<=-0.8 appears to be new relative to everything I could open.

## (i) Stable smooth branch vs a: published numbers

LSS, arXiv 2010.01201 (J. Nonlinear Sci. 31:82, 2021). Fetched PDF https://arxiv.org/pdf/2010.01201 (via local pdftotext). Stable branch only; a in [-5, 1]. Table 1 columns: gamma_e (nonlinear eigenvalue solve, only a>=-0.25 listed), then p, gamma (time-dependent fit), gamma2 (fit of max|omega_x|). Stated accuracy: gamma(a) 3-4 digits for -1<=a<=0.689; gamma_e 3-4 digits for a<0.3, >=5 digits for a>=0.3. CONFIRMED values (= c_l = 1+lambda):

| a | gamma_e (eigen) | gamma (time-dep) | gamma2 |
|---|---|---|---|
| -5 | - | 7.495 | 7.517 |
| -2 | - | 3.444 | 3.422 |
| -1 | - | 2.208 | 2.206 |
| -0.5 | - | 1.603747 | 1.600222 |
| -0.25 | 1.296593455 | 1.303708 | 1.302424 |
| -0.2 | 1.239824952 | 1.243558 | 1.242436 |
| -0.15 | 1.181358555 | 1.183300 | 1.182701 |
| -0.1 | 1.121312899 | 1.122630 | 1.122093 |
| -0.05 | 1.061051829 | 1.061617 | 1.061334 |
| 0 | 1 | 1.000243 | 1.000019 |
| 0.25 | 0.684513621 | 0.684650 | 0.684671 |
| 0.5 | 0.333333333 | 0.33333354 | 0.33333340 |
| 2/3 | 0.045170944220367 | | 0.04517094422035 |
| 0.6890665337007457 = a_c | 0 | | |

Notes:
- Not tabulated: a=-0.9, -0.8 (and any a in (-1,-0.25) other than -0.5). At a=-0.5 only the time-dependent fit exists: lambda about 0.6037 (gamma) / 0.6002 (gamma2), ~3-4 digits, with the two fits disagreeing at the 3rd digit.
- Eigenvalue solver (generalized Petviashvili) was not pushed below a=-0.2/-0.25: "Going below a = -0.2 is difficult ... more than 10^6 grid points" (CONFIRMED). So high-digit stable values exist only for a>=-0.25.
- Cross-check issue for you: LSS a=-1 gives c_l=2.208/2.206 (lambda 1.208/1.206) vs DeepMind lambda_0=1.18078 (c_l 2.18078): 2% apart, beyond LSS's claimed 3-4 digits. Time-dependent fits at a=-1 are evidently less accurate than claimed (my inference, UNCONFIRMED as to cause). Do not use LSS a<=-0.5 values as better than ~2 digits.
- Other a-dependent stable data: 2606.20467 (Agentic Symbolic Search) benchmark at a=0.25 from the HQWW fixed-point formulation: beta_ref = 0.6845136212507712 (CONFIRMED text; same as LSS 0.684513621). Their beta = our c_l. 2607.19762 recomputes c_l(a) on 0<a<a_c by Newton continuation (accuracy ~3e-4) and reports a_c to 0.04% (CONFIRMED, no table seen). GitHub Say43/PINN-Blowup-Profiles (PINN, a=0,0.25,0.5 only, low accuracy; tabulates 0.685, 0.335) is a hobby repo, low weight.
- 2201.06780: Fig. 8 plots lambda(a) for a in [-1,1] reproducing LSS, a_c ~ 0.6887; also gives lambda=1.1808 at a=-1 (CONFIRMED; figure itself not tabulated).
- HQWW (arXiv 2305.05895v5, https://arxiv.org/pdf/2305.05895, local pdftotext): proves existence of smooth/interiorly smooth positive-class profile for all a<=1 (CONFIRMED), conjectures uniqueness and continuity/monotonicity in a of the fixed point (CONFIRMED as "conjecture"), brackets a_c in about (0.5269, 0.7342). Not tabulated numbers of c_l(a). Existence/uniqueness is for the sign-definite class; they explicitly do not exclude sign-changing profiles (CONFIRMED).
- Exact values: a=0 c_l=1 (CLM, Omega=-2y/(1+y^2), confirmed in 2610.30492 eq. 14 and HQWW); a=1/2 c_l=1/3 (Chen; LSS) ; a=1 c_l=-1 (CHH). Small |a| series: Elgindi-Jeong (C^alpha); Zheng 2209.09886 abstract CONFIRMED: exactly self-similar C^alpha solutions for |a alpha| small, alpha in (0,1) (low regularity, c_l=1/alpha; not smooth).

## Singular (non-smooth) profiles for a<0, for contrast
2603.25104 Theorem 2.6 (CONFIRMED): for any a<0, an exact half-line singular profile with c_l = 1-a, c_omega=-1, i.e. our lambda = -a (a=-0.9 -> 0.9; a=-0.8 -> 0.8; a=-1 -> 1). This is a different (non-smooth, two-scale) object, but note it is roughly parallel to your lambda_1(a) (slope -1.04 vs -1), and sits between lambda_1 and lambda_0 at a=-0.9/-0.8. Probably coincidence, but worth a glance if your lambda_1 branch's profile develops a cusp as a increases.

## (iii) Where unstable branches end or merge

No paper states anything. CONFIRMED absent in all fetched texts. Related, loosely relevant facts:
- 2607.19762 Theorem 2 (CONFIRMED in abstract/intro text): at a=0 the full point spectrum of the CLM self-similar linearization on the odd realization is exactly {0,1} (the symmetry modes), no other eigenvalues. So at a=0 the CLM profile has no excited unstable modes. This is consistent with any CCF-type excited smooth branch having to terminate, turn singular, or leave the odd smooth class before a=0, but that is my inference, not a published claim. (2607.19762 s*(a)=1/c_l(a) dissipation exponent also uses only the stable branch.)
- LSS: a=0 and 1/2 are the only values with exact pole-type solutions (Theorem 3) and are special; no mention of excited branches.
- 2511.22819: at a=-1, no 3rd unstable solution found for lambda in [0.455, 0.4713] (non-smooth signal persists), so whether the CCF ladder continues is open. They give lambda_2 = 0.47132422 (one sentence in the intro says 0.4703, apparent typo; the figure caption and the scan range give 0.4713). Empirical relation 1/lambda_n linear in n is for IPM (and Boussinesq), eq. (11) fit a=1.014, b=1.059 is for another system; not applicable to CCF a-dependence.
- Elgindi-Jeong/Zheng: for a=0 there is a continuum of C^alpha (non-smooth) exact profiles with c_l=1/alpha (UNCONFIRMED formula detail beyond 2603.25104's quoted CLM family: Omega^(alpha) with gamma=1/alpha, CONFIRMED text). Only alpha=1 (lambda=0) is smooth. If unstable smooth branches die by losing smoothness near a=0, these are the natural neighbours; speculation.

## CCF values to compare (a=-1), CONFIRMED
- 2201.06780: lambda = 1.1808 (stable), 0.60573 (1st unstable); a=-1 ODIL/PINN benchmark values in 2610.30492: lambda_0 = 1.180777662899, lambda_1 = 0.6057337012032, lambda_2 = 0.471324863862 (the last differs from 2511.22819's 0.47132422 in the 7th digit; your 0.47132 agrees).
- 2509.14185 (https://arxiv.org/abs/2509.14185, HTML read): CCF stable and 1st unstable solved to near double precision; no a variation. 2511.22819: 2nd unstable.

## Citation sweep
Semantic Scholar via WebFetch returned 67 citers of 2201.06780 and 34 of 2509.14185 (2010.01201 not fetched). Relevant gCLM-type citers examined: 2603.25104, 2610.30492, 2511.22819, 2606.20467 (cites via DOI), 2607.19762 (from search), 2609.37232 (CCF Holder estimates, abstract only: CCF theta_t - u theta_x = 0 with theta_0=(1+x^2)^-1, no exponent info), 2411.01891 (exact periodic gCLM with dissipation at a=0,1/2; abstract/snippet only), 2401.14615 (multi-scale CLM blowups, a=0; not read in detail). None varies a for excited smooth branches. Not opened: Chen "slightly perturbed De Gregorio", Chen-Hou-Huang 1905.06387 (a=1, stable), Castro-Cordoba 2010 (a<0 blow-up existence), Eggers-Fontelos book (UNCONFIRMED: 2511.22819 attributes the first CCF unstable profile to Eggers-Fontelos, ref [8]/[31] "Singularities: formation, structure, and propagation", 2015; I did not verify this or look for a-dependence), Eggers-Fontelos Nonlinearity 33:325 (read abstract and header: fractional Burgers and Moore-type model, not gCLM with advection a; not relevant).

## URLs actually fetched
https://arxiv.org/abs/2010.01201 ; https://arxiv.org/pdf/2010.01201 ; https://arxiv.org/pdf/2305.05895 ; https://arxiv.org/abs/2603.25104 and /html/2603.25104 ; https://arxiv.org/html/{2201.06780, 2509.14185, 2511.22819, 2610.30492, 2607.19762, 2609.13220, 2606.20467, 2401.14615} ; https://arxiv.org/abs/{2209.08232, 2209.09886, 2609.37232, 2603.11301} ; https://ar5iv.labs.arxiv.org/html/0812.1339 ; https://people.maths.bris.ac.uk/~majge/Eggers_2020_Nonlinearity_33_325.pdf ; https://github.com/Say43/PINN-Blowup-Profiles ; Semantic Scholar citation API for 2201.06780 and 2509.14185.
