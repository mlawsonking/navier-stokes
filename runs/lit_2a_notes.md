# Literature notes 2a: CCF / IPM / Boussinesq self-similar exponents

Sources actually read:
- arXiv:2509.14185 PDF (https://arxiv.org/pdf/2509.14185, v1, local pdftotext, 20 pp; Supplementary Information NOT included, "to be uploaded in a forthcoming version"). Page text of https://arxiv.org/html/2509.14185 also fetched (no table, no CCF equation).
- arXiv:2201.06780 PDF (https://arxiv.org/pdf/2201.06780, main text + Supplementary; pdftotext DROPS ALL GREEK GLYPHS and formula symbols, so equations were reconstructed from surviving fragments plus https://ar5iv.labs.arxiv.org/html/2201.06780 which gave the main-text equation). arxiv.org/abs/2201.06780 fetched (abstract only).

## 1. CCF equation
- CONFIRMED (2201.06780 main text, generalized De Gregorio eq., p.4, and Supp. Sec 3.1 eq. before 3.1): omega_t + a u omega_x = u_x omega, with a = -1 being CCF. u = integral_0^x H(omega) (ar5iv: "u = int_0^x Hω = Λ^{-1}ω"), i.e. u_x = H omega.
- NOT FOUND / unreadable: sign and prefactor of H. In the PDF only "H = -1 ..." survives (the pi, the integral and the kernel were lost in extraction). Possibly H = -(1/pi) PV int omega(y)/(x-y)... but NOT verified; check the PDF visually. ar5iv says only "H is the Hilbert transform".
- 2509.14185 never writes the CCF equation (refers to ref. 4 CCF model, ref. 16 = Wang et al. 2201.06780; the equations are in the unavailable Supplementary Information).

## 2. Ansatz and profile ODE
- 2509.14185 Methods, "Self-similar Solutions", eq. (2): Phi_A(x,t) = (1-t)^{k_A(lambda)} Phi_A(y), y = (1-t)^{-(1+lambda)} x; blow-up at t=1; k_A linear in lambda; steady eqs E_k(Phi, lambda)=0 eq. (3). CONFIRMED (generic, not CCF-specific).
- 2201.06780 Supp. 3.1 / main p.4: omega = (1/(1-t)) Omega(x/(1-t)^{1+lambda}) (ar5iv confirms; extra slow time s=-log(1-t) in dissipative discussion). Amplitude exponent -1, spatial exponent 1+lambda: CONFIRMED. Profile ODE (Supp. eq 3.1), surviving fragments: Omega + ((1+lambda) y + a U) dOmega/dy - Omega dU/dy = 0, with dU/dy = H(Omega), U = Lambda^-1-type antiderivative so U(0)=0 (Supp. text "odd symmetry of U yields the implicit boundary condition U(0)=0"). The exact line is garbled; my consistency check: omega~(1-t)^-1, u~(1-t)^lambda (this u scaling is INFERRED, the PDF's "U = ...^{-1}" definition lost its exponent). For a=-1: Omega + ((1+lambda)y - U) Omega' = Omega U'.
- In sinh^{-1} coordinates y = sinh z (Supp. eq 3.4): Omega + ((1+lambda) sinh z + a U) sech z dOmega/dz - Omega sech z dU/dz... = 0 with sech z dU/dz = H(Omega) (garbled slightly).

## 3. Normalisation, symmetry, far field
- CONFIRMED (2201.06780 Supp. 3.1, eq 3.2 and loss 3.3): rescaling invariance Omega(y) -> c Omega(c y); fix dOmega/dy(0) = -2 (conditions g1 = Omega'(0)+2, g2 = U(|y|=d), g3 = Omega(|y|=d)). Omega and U both ODD (odd symmetry imposed via [NN(y)-NN(-y)]/2). Domain via y = sinh z, d up to sinh(30..40).
- For the unstable (2nd) CCF solution the paper switches normalisation: Omega(0.5) = -0.05 ("(0.5)=-0.05" Supp. 3.5; main text writes "= 0.05", sign lost in extraction) plus third-derivative smoothness constraint (3.8).
- Far field: Omega decays as a power law |y|^{-1/(1+lambda)} (Supp. 3.6: exponent "-1/(1+lambda)"). Main text footnote 14: gradient of U vanishes at infinity (Boussinesq).
- 2509.14185: hard-coded odd prefactor times power-law decay factor q = (1+|y|^2)^{-alpha/2}-type, alpha=1/(1+lambda) (eqs 5,6; written for Boussinesq vorticity). Normalisation: Methods eq. (13) point y_norm at high gradient, A0(y_norm)=C_norm; CCF-specific C_norm in SI (NOT FOUND).

## 4. CCF exponents
2509.14185 Fig. 2f table (columns CCF / IPM / Boussinesq-with-boundary; read from PDF text, p.4):
- Stable lam_0 = 1.1807776628998 (CONFIRMED, Fig 2f)
- 1st unstable lam_1 = 0.6057337012032 (Fig 2f); intro text p.4-5: lam_1 ~ 0.6057.
- 2nd unstable lam_2: intro text p.5 says 0.4703 ("lambda_2 = 0.4703", confirmed also in the HTML), BUT Fig 2f table lists 0.4713248638620. DISCREPANCY inside v1 (figure vs text); table has more digits and is consistent with the IPM/Boussinesq style; the text value may be a typo. Unresolved; need v2/SI or independent computation.
- Significant digits are bold-faced in the figure; bold is not visible in extracted text, so which digits are validated is UNKNOWN.
- 2201.06780: stable lambda ~ 1.1808 (Supp. 3.5), 1.18080 +/- 0.00005 (3.6), "1.18078" (main text p.6); unstable 0.60573 (main), 0.60573 +/- 0.00004 (3.6). Eigenvalue of unstable mode of 2nd CCF solution mu = 0.36525 > 0 (Supp. 3.7, Fig 12). Residuals ~1e-6 there; 2509.14185 residuals ~1e-13.
- Eggers and Fontelos (Nonlinearity 33, 325 (2019), ref 31 of 2509.14185) credited with the first unstable CCF profile; no numerical value quoted in either paper's text. Lushnikov-Silantyev-Siegel (J. Nonlinear Sci. 31 (2021)) and Chen-Hou-Huang: only cited as benchmarks; no CCF lambda quoted. NOT FOUND as numbers.
- Related: PINN a for lambda=-1 gives a ~ 0.6887 (2201.06780 Fig 8 caption; context garbled).
- Consequence for dissipative CCF: blow-up for alpha < 1/(1+lam): ~0.623 (lam_1), ~0.68 (lam_2=0.4703) (2509 p.5); 2201: alpha < ~0.61 (uses 1/(1+0.6057)=0.623 in 2509).

## 5. Analytic lambda relation
- CONFIRMED, 2509.14185 Methods, "Identifying Admissible lambda Values", method 2 "lambda Inference" (p.16): differentiate the self-similar equation once, evaluate at origin assuming smoothness, require vanishing -> explicit relation between lambda and origin field values. Quote: "the relation derived for the Boussinesq problem is that lambda = -3 - 2 d_{y1}U1(0,0), where d_{y1}U1(0,0) is the velocity gradient at the origin." Second-hand report CONFIRMED exactly. Used as an update rule during training. (Cites Prat-Colomer thesis.) Funnel inference (secant on max residual, eq 17-18) is method 3.
- CCF analogue not stated in the papers; but it follows trivially from the ODE: differentiating/evaluating Omega + ((1+lam)y + aU)Omega' - Omega U' = 0 at y=0 with U(0)=0, Omega(0)=0 (odd) gives, at first order, Omega'(0)[1 + (1+lam) + a U'(0) - U'(0)] = 0, i.e. lam = -2 + (1-a) U'(0) = -2 + 2U'(0) for CCF (a=-1), U'(0)=H(Omega)(0). THIS IS MY DERIVATION, NOT FROM THE PAPERS; verify numerically before use.

## 6. Boussinesq and IPM
- Fig 2f (2509.14185), IPM: stable 1.0285722760222, 1st 0.4721297362414, 2nd 0.3149620267088, 3rd 0.2415604743989. Boussinesq (with boundary): stable 1.9205599746927, 1st 1.3990961221852, 2nd 1.2523481636489, 3rd 1.1842500861997, 4th 1.1448833556188 (4th un-validated, per caption). CONFIRMED (column assignment from layout; IPM column has no 4th entry).
- Boussinesq ansatz/decay: omega=curl U ~ |y|^{-alpha}, alpha=1/(1+lambda) (Methods p.13). Hard-coded odd-in-y1 vorticity ansatz eq (6): omega = y1/(1+y1^2+y2^2) NN[q,theta] q.
- Boussinesq/IPM profile equations: NOT FOUND (in Supplementary Information, absent). 2201.06780 states Boussinesq stable lambda ~ 1.9 (main text p.4, "exponent ~1.9"); its Boussinesq profile eqs (eq 3 there) were not extracted (symbols lost).
- 2509.14185 residual table: CCF stable -13.714, 1st -13.589, 2nd -6.664 (log10 max residual, Fig 3b).

## Open items for a human with the PDF open
Exact sign/prefactor of H; exact printed form of ODE 3.1; the sign in Omega(0.5)=+/-0.05; which Fig 2f digits are bold; whether 0.4703 or 0.4713248638620 is correct for lam_2.
