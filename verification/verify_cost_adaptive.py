"""Cost-aware two-stage adaptive RSS/SRS design: full simulation (m = 5, N(0,1) base model).

Cost model : 1 per measured unit, gamma per ranked set; total budget C.
Strategies : always-SRS (C units); always-RSS (kR = floor(C/(m(1+gamma))) cycles);
             adaptive  = RSS pilot of k0 cycles, efficiency estimated from the pilot itself,
                         continue RSS if RE_hat > 1+gamma, otherwise spend the rest on SRS and apply the
                         estimator to the pooled pilot + SRS sample (for the mean: weights n0/N_S, n2/N_S).
Common random numbers: the always-RSS sample is the pilot plus its RSS continuation, and the
adaptive SRS branch uses the first n2S units of the always-SRS sample (independent of the pilot).
Requires P_mech.npz written by verify_ordering.py.  Output: cost_adaptive_results.json
"""
import numpy as np, json, sys, time
from scipy import stats, integrate
from math import comb
m=5; CH=1.345; EPS=0.10
Pm=np.load("P_mech.npz")
MECH={"random":np.full((m,m),1/m)}
for a,b in(("G","Gaussian"),("L","Laplace")):
    for t,k in((.25,"0.25"),(.5,"0.5"),(.75,"0.75"),(.9,"0.9")): MECH["%s %.2f"%(b,t)]=Pm["%s_%s"%(a,k)]
MECH["perfect"]=np.eye(m)
e=np.array([integrate.quad(lambda x:x*r*comb(m,r)*stats.norm.cdf(x)**(r-1)*stats.norm.sf(x)**(m-r)*stats.norm.pdf(x),-12,12)[0] for r in range(1,m+1)])
def contaminate(x,rng):
    hit=rng.random(x.shape)<EPS
    return np.where(hit,8.0*rng.choice([-1.0,1.0],size=x.shape),x)
def rss(P,reps,k,cont,rng):
    """(reps,k,m): entry [.,.,s] is the measured unit of observed rank s+1."""
    cum=np.cumsum(P,axis=0)                               # column s: law of the true rank given observed rank s
    r=np.minimum((rng.random((reps,k,m))[...,None]>cum.T[None,None]).sum(-1),m-1)   # 0-based true rank
    if cont=="pre":                                       # contaminated values are present when the set is ranked
        sets=np.sort(contaminate(rng.standard_normal((reps,k,m,m)),rng),axis=-1)
        return np.take_along_axis(sets,r[...,None],axis=-1)[...,0]
    x=stats.norm.ppf(rng.beta(r+1,m-r))
    return contaminate(x,rng) if cont=="post" else x
def srs(reps,n,cont,rng):
    x=rng.standard_normal((reps,n)); return x if cont=="clean" else contaminate(x,rng)
def huber(x):
    th=np.median(x,axis=1)
    for _ in range(30):
        u=x-th[:,None]; th=th+np.clip(u,-CH,CH).mean(1)/np.maximum((np.abs(u)<CH).mean(1),1e-8)
    return th
def trimmed(x):
    n=x.shape[1]; g=int(np.floor(0.10*n)); return np.sort(x,axis=1)[:,g:n-g].mean(1)
EST={"mean":lambda x:x.mean(1),"trimmed":trimmed,"huber":huber}
def re_hat(h,k0):
    """h: (reps,k0,m) scores of the pilot. Unbiased within/between decomposition by observed-rank stratum."""
    W=h.var(axis=1,ddof=1).mean(1); sm=h.mean(1)
    B=((sm-sm.mean(1,keepdims=True))**2).mean(1)-(m-1)*W/(m*k0)
    return np.maximum((W+B)/np.maximum(W,1e-300),1.0)
def scores(name,pil):
    flat=pil.reshape(pil.shape[0],-1)
    if name=="mean": return pil
    if name=="huber": return np.clip(pil-huber(flat)[:,None,None],-CH,CH)
    lo,hi=np.quantile(flat,[0.10,0.90],axis=1); return np.clip(pil,lo[:,None,None],hi[:,None,None])
def run(P,cont,gam,k0,C,reps,rng,chunk=10000):
    kR=int(np.floor(C/(m*(1+gam))+1e-9)); n0=k0*m; n2S=int(np.floor(C-n0*(1+gam)+1e-9)); wS=n0/(n0+n2S)
    acc={k:np.zeros(4) for k in EST}; done=0
    while done<reps:
        b=min(chunk,reps-done); done+=b
        R=rss(P,b,kR,cont,rng); S=srs(b,C,cont,rng); pil=R[:,:k0,:]
        for name,f in EST.items():
            tS=f(S); tR=f(R.reshape(b,-1))
            go=re_hat(scores(name,pil),k0)>1+gam
            tA=np.where(go,tR,f(np.concatenate([pil.reshape(b,-1),S[:,:n2S]],axis=1)))   # SRS branch: estimator on the pooled sample
            acc[name]+=np.array([(tS**2).sum(),(tR**2).sum(),(tA**2).sum(),go.sum()])
    return {k:dict(mseS=v[0]/reps,mseR=v[1]/reps,mseA=v[2]/reps,pR=v[3]/reps,kR=kR,n2S=n2S) for k,v in acc.items()}
if __name__=="__main__":
    REPS=int(sys.argv[1]) if len(sys.argv)>1 else 40000
    rng=np.random.default_rng(20261004); out=[]; t0=time.time()
    for gam in(0.10,0.25,0.50):
        for cont in("clean","post","pre"):
            for name,P in MECH.items():
                res=run(P,cont,gam,20,500,REPS,rng)
                out.append(dict(part="main",gamma=gam,cont=cont,mech=name,k0=20,C=500,RE=float(m/(m-np.sum((P.T@e)**2))),res=res))
        print("gamma",gam,"done %.0fs"%(time.time()-t0),flush=True)
    for k0 in(5,10,20,40):                                   # pilot-size sensitivity (clean)
        for name,P in MECH.items():
            out.append(dict(part="pilot",gamma=0.25,cont="clean",mech=name,k0=k0,C=500,RE=float(m/(m-np.sum((P.T@e)**2))),res=run(P,"clean",0.25,k0,500,REPS,rng)))
    for C in(250,1000):                                      # budget sensitivity (clean)
        for name,P in MECH.items():
            out.append(dict(part="budget",gamma=0.25,cont="clean",mech=name,k0=20,C=C,RE=float(m/(m-np.sum((P.T@e)**2))),res=run(P,"clean",0.25,20,C,REPS,rng)))
    json.dump(dict(reps=REPS,rows=out),open("cost_adaptive_results.json","w"),indent=1)
    print("total %.0fs"%(time.time()-t0))
