import torch, math, time
from gpu_solver import ccf_sing as S, ccf_fast as F
DEV="cuda"; sols=torch.load("runs/sing_track_t4.pt"); print([round(x['lam'],3) for x in sols])
shape=(384,8,4)
for lam_t in (0.52, 0.50):
    x=[t for t in sols if abs(t['lam']-lam_t)<1e-9][0]; p=x['p'].to(DEV); F.MAP=tuple(x['map'])
    yy=torch.exp(torch.linspace(math.log(0.3),math.log(30.),60001,dtype=S.DT,device=DEV))
    Om=(S._fields(p[0],p[1],yy,shape)[1]@p[2:]).abs(); i=Om.argmax(); hw=(Om>Om[i]/2).nonzero().flatten()
    xp=math.log(yy[i].item()); fwxi=math.log(yy[hw[-1]].item()/yy[hw[0]].item()); print("lam",lam_t,"peak xi",xp,"fwhm in xi",fwxi)
    for cl in (None,(xp,0.5,0.45),(xp,0.3,0.285)):
        G=S.EGrid(h=0.00625,device=DEV,cluster=cl); t=time.time()
        q,h=S.lm(p,G,shape,max_iter=60); c=q[2:386].abs()
        print("  cluster",cl,"mu",q[1].item(),"sup",h[-1][1] if h else None,"its",len(h),"coef",[f"{c[k:k+48].max().item():.0e}" for k in range(0,384,48)],f"{time.time()-t:.0f}s",flush=True)
        del G; torch.cuda.empty_cache()
