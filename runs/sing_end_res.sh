#!/bin/bash
# resolution / scale study of the A_R = A_L member (see sing_end_rho1.py)
cd /c/Users/bobdo/projects/navier-stokes && export PYTHONPATH=.
NB=10 NFAR=16 .venv/Scripts/python -u runs/sing_end_rho1.py q5 320 160 0.4535843157 0.4535843160 runs/sing_end_q4_5.pt 5 2
NB=6 NFAR=10 .venv/Scripts/python -u runs/sing_end_rho1.py q3 192 96 0.4535843157 0.4535843160 runs/sing_end_q4_5.pt
