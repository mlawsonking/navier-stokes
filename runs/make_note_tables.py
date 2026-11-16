"""Markdown tables for NOTE.md from the run ledgers (nothing is recomputed)."""
import json
R = [json.loads(l) for l in open("runs/gclm_u1_branch.jsonl")]
trust = ("upe256", "upf256", "upg256", "uph256", "upl256", "log-w", "down192b", "verify")
R = [r for r in R if any(t in r["case"] for t in trust) and "lam" in r]
print("| a | lam_1 | lam_1 predicted -2(1+a)/(5-3a) | peak of abs(Omega) | residual (off-grid where recorded) | run |\n|---|---|---|---|---|---|")
for a in (-1.36, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.435, -0.4, -0.3, -0.2, -0.15, -0.1, -0.09, -0.08, -0.0712):
    c = sorted(R, key=lambda r: (round(abs(r["a"] - a), 4), "res_offgrid" not in r, r.get("res_offgrid", r.get("res_sup", 1))))[0]
    res = c.get("res_offgrid", c.get("res_sup")); pk = c.get("peak")
    print(f"| {c['a']:+.4f} | {c['lam']:+.12f} | {-2*(1+c['a'])/(5-3*c['a']):+.4f} | {pk if pk is None else round(pk, 3)} | {res:.1e}{'' if 'res_offgrid' in c else ' (on-grid only)'} | {c['case'].replace('gclm-u1-', '')} |")
S = [json.loads(l) for l in open("runs/sing_track.jsonl")]; S = [r for r in S if "res_offgrid" in r and "mu" in r]
print("\n| lam | mu | U'(0) | peak of abs(Omega) | off-grid residual | run |\n|---|---|---|---|---|---|")
for l in (0.60, 0.58, 0.56, 0.54, 0.53, 0.52, 0.50, 0.49, 0.48, 0.4713, 0.468, 0.465, 0.462, 0.46, 0.4594, 0.458, 0.457, 0.4559, 0.455, 0.4548, 0.45454):
    c = sorted(S, key=lambda r: (round(abs(r["lam"] - l), 5), r["res_offgrid"]))[0]
    print(f"| {c['lam']:.6f} | {c['mu']:.5f} | {c.get('up0', float('nan')):.6f} | {c['peak']:.3f} | {c['res_offgrid']:.1e} | {c['case'].replace('sing-', '')} |")
