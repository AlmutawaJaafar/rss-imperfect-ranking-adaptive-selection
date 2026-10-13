"""Real-data illustration: abalone data (Nash et al., 1994; UCI Machine Learning Repository), N = 4177.
Response: number of shell rings (age; obtained by cutting, staining and counting under a microscope).
Ranking variables: length, height, whole weight (cheap, non-destructive). Sets of m = 5 are drawn with
replacement from the data, ranked on the ranking variable (ties at random), and the rings of the selected unit are 'measured'.
Requires abalone.csv (columns: sex,length,diameter,height,whole,shucked,viscera,shell,rings; no header)."""
import numpy as np, json, sys
from scipy import stats
m=5; C=500; K0={0.10:14,0.25:11,0.50:8}; REPS=int(sys.argv[1]) if len(sys.argv)>1 else 40000
d=np.genfromtxt("abalone.csv",delimiter=",",dtype=None,encoding="utf8")
col={n:i for i,n in enumerate(["sex","length","diameter","height","whole","shucked","viscera","shell","rings"])}
y=np.array([r[col["rings"]] for r in d],float); N=len(y); mu=y.mean(); var=y.var()
rng=np.random.default_rng(4177)
def draw_sets(x,shape):                 # index of the unit of observed rank s (ranking on x, ties at random) in each set
    idx=rng.integers(0,N,size=shape+(m,)); key=x[idx]+1e-9*rng.random(idx.shape)
    return idx,np.argsort(key,axis=-1)
def rss(x,reps,k):                       # (reps,k,m): set s of a cycle contributes its unit of observed rank s
    idx,order=draw_sets(x,(reps,k,m)); pick=np.take_along_axis(order,np.arange(m)[None,None,:,None],axis=-1)[...,0]
    return y[np.take_along_axis(idx,pick[...,None],axis=-1)[...,0]]
def re_hat(h,k0):
    W=h.var(axis=1,ddof=1).mean(1); sm=h.mean(1); B=((sm-sm.mean(1,keepdims=True))**2).mean(1)-(m-1)*W/(m*k0)
    return np.maximum((W+B)/np.maximum(W,1e-300),1.0),W,B,sm
out={"N":N,"mean":mu,"var":var,"rankers":{}}
print("N %d, mean rings %.3f, variance %.3f"%(N,mu,var))
rr=np.arange(1,m+1)-3.0
for name in("length","height","whole"):
    x=np.array([r[col[name]] for r in d],float)
    # population quantities from 2,000,000 sets
    idx,order=draw_sets(x,(2_000_000,)); ys=y[idx]
    yobs=np.take_along_axis(ys,order,axis=-1)                               # column s: rings of the unit of observed rank s
    mus=yobs.mean(0); vs=yobs.var(0); RE=var/vs.mean()
    ytrue=np.argsort(np.argsort(ys+1e-9*rng.random(ys.shape),axis=-1),axis=-1)   # true rank (ties at random) of each unit
    tr_of_obs=np.take_along_axis(ytrue,order,axis=-1)                       # true rank of the unit of observed rank s
    P=np.array([[np.mean(tr_of_obs[:,s]==r) for s in range(m)] for r in range(m)])
    e=np.sort(ys,axis=-1).mean(0)-mu; rhoR=rr@P@rr/(rr@rr); RE_rankonly=m*var/(m*var-np.sum((P.T@e)**2)); REperf=m*var/(m*var-e@e)
    Hb=np.linalg.svd(np.eye(m)-np.ones((m,m))/m)[0][:,:m-1]; lam=np.linalg.eigvalsh(Hb.T@P@P.T@Hb)
    print("\n%s: Spearman %.3f | rho_R %.3f | RE %.3f (rank-only prediction %.3f; perfect ranking %.3f) | lambda_min %.3f lambda_max %.3f"%(name,stats.spearmanr(x,y)[0],rhoR,RE,RE_rankonly,REperf,lam[0],lam[-1]))
    print("   stratum means",np.round(mus,2))
    res={"spearman":float(stats.spearmanr(x,y)[0]),"rhoR":float(rhoR),"RE":float(RE),"RE_rankonly":float(RE_rankonly),"RE_perfect":float(REperf),"design":{}}
    for gam,k0 in K0.items():
        kR=int(np.floor(C/(m*(1+gam))+1e-9)); n0=k0*m; n2=int(np.floor(C-n0*(1+gam)+1e-9)); acc=np.zeros(4); done=0
        while done<REPS:
            b=min(5000,REPS-done); done+=b
            R=rss(x,b,kR); S=y[rng.integers(0,N,size=(b,C))]; pil=R[:,:k0,:]
            go=re_hat(pil,k0)[0]>1+gam
            tS=S.mean(1); tR=R.mean((1,2)); tA=np.where(go,tR,(pil.sum((1,2))+S[:,:n2].sum(1))/(n0+n2))
            acc+=np.array([((tS-mu)**2).sum(),((tR-mu)**2).sum(),((tA-mu)**2).sum(),go.sum()])
        mS,mR,mA,pR=acc/REPS; vS,vR=var/C,var/(kR*m*RE); o=min(vS,vR)
        print("   gamma %.2f (k0 %2d): cost-adjusted efficiency %.3f | Pr(RSS) %.3f | regret %%: SRS %.1f RSS %.1f adaptive %.1f"%(gam,k0,RE*kR*m/C,pR,100*(vS/o-1),100*(vR/o-1),100*(mA/o-1)))
        res["design"][str(gam)]=dict(k0=k0,costadj=float(RE*kR*m/C),pR=float(pR),regS=float(100*(vS/o-1)),regR=float(100*(vR/o-1)),regA=float(100*(mA/o-1)),simS=float(mS),simR=float(mR))
    out["rankers"][name]=res
# one realized pilot, whole weight, gamma = 0.25
x=np.array([r[col["whole"]] for r in d],float); rng=np.random.default_rng(2026)
pil=rss(x,1,11); REh,W,B,sm=re_hat(pil,11); F=11*((sm-sm.mean())**2).sum()/(m-1)/W[0]
print("\none pilot (whole weight, 11 cycles): stratum means",np.round(sm[0],2),"W %.2f B %.2f  F %.2f  RE_hat %.3f  threshold 1.25 ->"%(W[0],B[0],F,REh[0]),"RSS" if REh[0]>1.25 else "SRS")
out["pilot_example"]=dict(stratum_means=sm[0].tolist(),W=float(W[0]),B=float(B[0]),F=float(F),RE_hat=float(REh[0]))
json.dump(out,open("abalone_results.json","w"),indent=1)
