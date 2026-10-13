"""Additional experiments for the cost-aware adaptive design.
   part = setsize3 | setsize7 | lognormal | asym | misspec      (usage: python3 verify_extra_experiments.py <part> [reps])
   setsize*  : standard normal, m = 3 or 7, ten mechanisms (calibrated here), clean / post / pre symmetric contamination
   lognormal : lognormal(0,1) population, m = 5, clean, same ten mechanisms as the main experiment
   asym      : standard normal, m = 5, 10% contamination at +8 (prob 0.75) or -8 (prob 0.25), post and pre
   misspec   : standard normal mean, m = 5, gamma = 0.25, decision threshold 1+gamma_hat with gamma_hat != gamma
   Budget C = 500; pilot sizes are the approximately minimax ones for the normal mean. Output: extra_<part>.json"""
import numpy as np, json, sys, time
from scipy import stats, integrate, optimize
from math import comb
CH=1.345; EPS=0.10; C=500
part=sys.argv[1]; REPS=int(sys.argv[2]) if len(sys.argv)>2 else 40000
m={"setsize3":3,"setsize7":7}.get(part,5)
K0={3:{0.10:23,0.25:16,0.50:12},5:{0.10:14,0.25:11,0.50:8},7:{0.10:10,0.25:8,0.50:6}}[m]
POP="lognormal" if part=="lognormal" else "normal"; PPLUS=0.75 if part=="asym" else 0.5
tr=(lambda z:np.exp(z)) if POP=="lognormal" else (lambda z:z)
J=np.eye(m)[::-1]; rr=np.arange(1,m+1)-(m+1)/2
def osmean(f): return np.array([integrate.quad(lambda x:f(x)*r*comb(m,r)*stats.norm.cdf(x)**(r-1)*stats.norm.sf(x)**(m-r)*stats.norm.pdf(x),-12,12,epsabs=1e-12)[0] for r in range(1,m+1)])
ev=osmean(tr); mu=ev.mean(); e=ev-mu; var=integrate.quad(lambda x:(tr(x)-mu)**2*stats.norm.pdf(x),-12,12)[0]
def REm(P): return m*var/(m*var-np.sum((P.T@e)**2))
def contaminate(x,rng):
    hit=rng.random(x.shape)<EPS; return np.where(hit,8.0*np.where(rng.random(x.shape)<PPLUS,1.0,-1.0),x)
def rss(P,reps,k,cont,rng):
    cum=np.cumsum(P,axis=0); r=np.minimum((rng.random((reps,k,m))[...,None]>cum.T[None,None]).sum(-1),m-1)
    if cont=="pre":
        sets=np.sort(contaminate(rng.standard_normal((reps,k,m,m)),rng),axis=-1); return np.take_along_axis(sets,r[...,None],axis=-1)[...,0]
    x=tr(stats.norm.ppf(rng.beta(r+1,m-r))); return contaminate(x,rng) if cont=="post" else x
def srs(reps,n,cont,rng):
    x=tr(rng.standard_normal((reps,n))); return x if cont=="clean" else contaminate(x,rng)
def huber(x):
    th=np.median(x,axis=1)
    for _ in range(40):
        u=x-th[:,None]; th=th+np.clip(u,-CH,CH).mean(1)/np.maximum((np.abs(u)<CH).mean(1),1e-8)
    return th
def trimmed(x):
    n=x.shape[1]; g=int(np.floor(0.10*n)); return np.sort(x,axis=1)[:,g:n-g].mean(1)
EST={"mean":lambda x:x.mean(1),"trimmed":trimmed,"huber":huber}
if part=="misspec": EST={"mean":EST["mean"]}
def re_hat(h,k0):
    W=h.var(axis=1,ddof=1).mean(1); sm=h.mean(1); B=((sm-sm.mean(1,keepdims=True))**2).mean(1)-(m-1)*W/(m*k0)
    return np.maximum((W+B)/np.maximum(W,1e-300),1.0)
def scores(name,pil):
    flat=pil.reshape(pil.shape[0],-1)
    if name=="mean": return pil
    if name=="huber": return np.clip(pil-huber(flat)[:,None,None],-CH,CH)
    lo,hi=np.quantile(flat,[0.10,0.90],axis=1); return np.clip(pil,lo[:,None,None],hi[:,None,None])
def run(P,cont,gam,k0,reps,rng,gam_hat=None):
    chunk=5000 if (cont=="pre" and m==7) else 10000
    kR=int(np.floor(C/(m*(1+gam))+1e-9)); n0=k0*m; n2S=int(np.floor(C-n0*(1+gam)+1e-9)); thr=1+(gam if gam_hat is None else gam_hat)
    acc={k:np.zeros(7) for k in EST}; done=0
    while done<reps:
        b=min(chunk,reps-done); done+=b
        R=rss(P,b,kR,cont,rng); S=srs(b,C,cont,rng); pil=R[:,:k0,:]
        for name,f in EST.items():
            tS=f(S); tR=f(R.reshape(b,-1)); go=re_hat(scores(name,pil),k0)>thr
            tA=np.where(go,tR,f(np.concatenate([pil.reshape(b,-1),S[:,:n2S]],axis=1)))
            acc[name]+=np.array([tS.sum(),(tS**2).sum(),tR.sum(),(tR**2).sum(),tA.sum(),(tA**2).sum(),go.sum()])
    return {k:dict(zip(("mS","qS","mR","qR","mA","qA","pR"),(v/reps).tolist()),kR=kR,n2S=n2S) for k,v in acc.items()}
def mechanisms(rng):
    M={"random":np.full((m,m),1/m)}
    if m==5:
        Pm=np.load("P_mech.npz")
        for a,b in(("G","Gaussian"),("L","Laplace")):
            for t,k in((.25,"0.25"),(.5,"0.5"),(.75,"0.75"),(.9,"0.9")): M["%s %.2f"%(b,t)]=Pm["%s_%s"%(a,k)]
    else:                                                    # calibrate tau on common random numbers so that the realized rho_R equals the target
        N=2_000_000; Z={"Gaussian":rng.standard_normal((N,m)),"Laplace":rng.laplace(size=(N,m))}
        def Pof(z,tau):
            obs=np.argsort(np.argsort(np.arange(1,m+1)+tau*z,axis=1),axis=1)
            P=np.array([np.bincount(obs[:,i],minlength=m) for i in range(m)])/N; return (P+J@P@J)/2
        for name,z in Z.items():
            for t in(.25,.5,.75,.9):
                tau=optimize.brentq(lambda s:rr@Pof(z,s)@rr/(rr@rr)-t,0.02,60,xtol=1e-5); M["%s %.2f"%(name,t)]=Pof(z,tau); M["tau %s %.2f"%(name,t)]=tau
    M["perfect"]=np.eye(m); return M
def targets():
    """functional values T(F) of the three estimators for each contamination setting"""
    T={}
    if POP=="lognormal":
        T["clean"]=dict(mean=mu,trimmed=integrate.quad(lambda u:np.exp(stats.norm.ppf(u)),0.1,0.9)[0]/0.8,
                        huber=optimize.brentq(lambda th:integrate.quad(lambda z:np.clip(np.exp(z)-th,-CH,CH)*stats.norm.pdf(z),-12,12)[0],0.1,5))
    elif part=="asym":
        lo,hi=EPS*(1-PPLUS),1-EPS*PPLUS
        tm=integrate.quad(lambda u:stats.norm.ppf((u-lo)/(1-EPS)),0.1,0.9)[0]/0.8
        hb=optimize.brentq(lambda th:(1-EPS)*integrate.quad(lambda z:np.clip(z-th,-CH,CH)*stats.norm.pdf(z),-12,12)[0]+EPS*PPLUS*min(8-th,CH)+EPS*(1-PPLUS)*max(-8-th,-CH),-1,2)
        for c in("post","pre"): T[c]=dict(mean=8*EPS*(2*PPLUS-1),trimmed=tm,huber=hb)
    else:
        for c in("clean","post","pre"): T[c]=dict(mean=0.0,trimmed=0.0,huber=0.0)
    return T
if __name__=="__main__":
    rng=np.random.default_rng({"setsize3":303,"setsize7":707,"lognormal":505,"asym":606,"misspec":808}[part]); t0=time.time()
    M=mechanisms(rng); taus={k:v for k,v in M.items() if k.startswith("tau")}; M={k:v for k,v in M.items() if not k.startswith("tau")}
    conts={"lognormal":("clean",),"asym":("post","pre"),"misspec":("clean",)}.get(part,("clean","post","pre"))
    out=[]
    if part=="misspec":
        for f in(0.5,0.75,1.0,1.5,2.0):
            for name,P in M.items(): out.append(dict(gamma=0.25,gamma_hat=0.25*f,cont="clean",mech=name,k0=11,RE=float(REm(P)),res=run(P,"clean",0.25,11,REPS,rng,0.25*f)))
            print("gamma_hat/gamma",f,"done %.0fs"%(time.time()-t0),flush=True)
    else:
        for gam in(0.10,0.25,0.50):
            for cont in conts:
                for name,P in M.items(): out.append(dict(gamma=gam,cont=cont,mech=name,k0=K0[gam],RE=float(REm(P)),res=run(P,cont,gam,K0[gam],REPS,rng)))
            print("gamma",gam,"done %.0fs"%(time.time()-t0),flush=True)
    json.dump(dict(part=part,m=m,reps=REPS,C=C,var=var,targets=targets(),taus=taus,rows=out),open("extra_%s.json"%part,"w"),indent=1)
    print("total %.0fs"%(time.time()-t0))
