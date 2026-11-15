import sys, math, torch
from gpu_solver import ccf_sing as S, ccf_fast as F, gclm_log as L
DEV="cuda"; x=torch.load("runs/ccf_u2_smooth640.pt"); p=x["p"].to(DEV); mp=tuple(x["map"]); F.MAP=mp; n0=640; lam=p[0]; L.D=S.D
def nodes(n):
    j=torch.arange(n,dtype=S.DT,device=DEV); t=torch.cos(math.pi*(j+0.5)/n); x0,eps=mp; b1,b2=math.asinh((1-x0)/eps),math.asinh((1+x0)/eps)
    return (x0+eps*torch.sinh((b1+b2)*(t+1)/2-b2)+1)/2
zero=torch.zeros(1,dtype=S.DT,device=DEV)
M=lambda s:(S.E2(s,1-s,p[1],(n0,1,0))[0]@p[2:]); M0=M(zero)[0]; beta=lam/(2*(1+lam))
G=S.peak_grid(lam.item(),7.722,mp,640,h=0.0125,R=6,device=DEV)
rs=S.residual(p,G,(n0,1,0)); print("ccf_sing residual on this grid", rs.abs().max().item(), "rows", rs.numel())
for n in (448,640):
    s=nodes(n); q=torch.cat([lam.reshape(1),(2*beta*M0).reshape(1),torch.linalg.solve(F.basis3(s,n)[0],(M(s)-M0)/s)])
    r=L.residual(q[0],q[1:],G,-1.0); i=r[:-2].abs().argmax(); print(n,"gclm_log start sup",r.abs().max().item(),"at y",G.y[i].item(),"last rows",r[-2:].tolist(), "c_last", q[-4:].abs().max().item())
    # fields agreement
    Om_s=S._fields(p[0],p[1],G.y,(n0,1,0))[1]@p[2:]; Om_l=L._basis(q[0],G.y,n)[1]@q[1:]
    print("  Omega diff sup",(Om_s-Om_l).abs().max().item(),"Omega max",Om_s.abs().max().item())
    q2,hist=L.gn(q,G,a=-1.0,max_iter=12,verbose=True)
