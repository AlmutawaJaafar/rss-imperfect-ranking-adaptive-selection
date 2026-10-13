import numpy as np, json, sys
sys.argv=["x"]; exec(open("verify_cost_adaptive.py").read().split('if __name__')[0])
d=json.load(open("cost_adaptive_results.json")); rows=[r for r in d["rows"] if r["part"]=="main"]
pred={("0.25","post"):(1.004,1.028,1.030),("0.25","pre"):(1.026,1.043,1.043),("0.50","post"):(1.018,1.124,1.132),("0.50","pre"):(1.113,1.199,1.199),
      ("0.75","post"):(1.041,1.331,1.356),("0.75","pre"):(1.293,1.595,1.597),("0.90","post"):(1.061,1.556,1.607),("0.90","pre"):(1.488,2.157,2.163),
      ("0.25","clean"):(1.041,),("0.50","clean"):(1.190,),("0.75","clean"):(1.556,),("0.90","clean"):(2.059,)}
print("(D) i.i.d. contamination: predicted vs observed per-unit efficiency (avg over 2 mechanisms x 3 budgets; s.e. ~ RE*0.0041)")
inside=0;tot=0
for (rho,cont),pv in pred.items():
    for est,p in zip(("mean","trimmed","huber"),pv):
        v=[r["res"][est]["mseS"]*500/(r["res"][est]["mseR"]*r["res"][est]["kR"]*m) for r in rows if r["cont"]==cont and r["mech"].endswith(rho) ]
        o=np.mean(v); se=o*np.sqrt(4/40000)/np.sqrt(len(v)); tot+=1; inside+=abs(o-p)<1.96*se
        print("%s %-5s %-8s pred %.3f obs %.3f  diff %+.3f  z %+.1f (n=%d)"%(rho,cont,est,p,o,o-p,(o-p)/se,len(v)))
print("inside 95%%: %d of %d"%(inside,tot))
# (E) pooled-sample variant in the SRS branch
def run2(P,cont,gam,k0,C,reps,rng,chunk=10000):
    kR=int(np.floor(C/(m*(1+gam))+1e-9)); n0=k0*m; n2S=int(np.floor(C-n0*(1+gam)+1e-9)); wS=n0/(n0+n2S)
    acc={k:np.zeros(4) for k in EST}; done=0
    while done<reps:
        b=min(chunk,reps-done); done+=b
        R=rss(P,b,kR,cont,rng); S=srs(b,C,cont,rng); pil=R[:,:k0,:]
        for name,f in EST.items():
            tS=f(S); tR=f(R.reshape(b,-1)); go=re_hat(scores(name,pil),k0)>1+gam
            tW=np.where(go,tR,wS*f(pil.reshape(b,-1))+(1-wS)*f(S[:,:n2S]))
            tP=np.where(go,tR,f(np.concatenate([pil.reshape(b,-1),S[:,:n2S]],axis=1)))
            acc[name]+=np.array([(tS**2).sum(),(tR**2).sum(),(tW**2).sum(),(tP**2).sum()])
    return {k:v/reps for k,v in acc.items()}
rng=np.random.default_rng(5)
print("\n(E) gamma 0.25: adaptive regret (%) with weighted combination vs pooled sample in the SRS branch")
res={}
for cont in("clean","post","pre"):
    for name in("random","Gaussian 0.25","Gaussian 0.50","Gaussian 0.75"):
        r=run2(MECH[name],cont,0.25,20,500,40000,rng)
        for est in("mean","trimmed","huber"):
            v=r[est]; o=min(v[0],v[1]); res[(cont,name,est)]=(100*(v[2]/o-1),100*(v[3]/o-1))
        print(cont,"%-14s"%name," ".join("%s: weighted %.1f pooled %.1f |"%(est[:4],*res[(cont,name,est)]) for est in("mean","trimmed","huber")))
for est in("mean","trimmed","huber"):
    print(est,"max weighted %.1f max pooled %.1f"%(max(v[0] for k,v in res.items() if k[2]==est),max(v[1] for k,v in res.items() if k[2]==est)))
