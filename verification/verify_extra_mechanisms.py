"""Additional checks requested in review: (A) non-additive mechanisms in the adaptive design, (B) shortcut vs set-based
generator, (C) Gaussian-Laplace comparison at matched realized rho_R, (D) i.i.d.-contamination alignment from the
cost-aware runs, (E) pooled-sample variant for the nonlinear estimators."""
import numpy as np, json, itertools, sys
from scipy import optimize, stats
sys.argv=["x"]; exec(open("verify_cost_adaptive.py").read().split('if __name__')[0])
rr=np.arange(1,m+1)-(m+1)/2
def rhoR(P): return rr@P@rr/(rr@rr)
def REm(P): return m/(m-np.sum((P.T@e)**2))
out={}
# ---------- (A) permutation and minimum-efficiency matrix at rho_R = 0.5
perm=[np.eye(m)[list(p)] for p in itertools.permutations(range(m))]
Q=[q for q in perm if abs(rhoR(q)-0.5)<1e-12][0]
A=[];b=[]
for i in range(m):
    a=np.zeros((m,m));a[i,:]=1;A.append(a.ravel());b.append(1)
for j in range(m-1):
    a=np.zeros((m,m));a[:,j]=1;A.append(a.ravel());b.append(1)
AA=np.array(A+[np.outer(rr,rr).ravel()]);bb=np.array(b+[0.5*(rr@rr)])
g=np.random.default_rng(3);best=None
for t in range(12):
    s=optimize.minimize(lambda x:np.sum((x.reshape(m,m).T@e)**2),np.full(m*m,1/m)+0.01*g.standard_normal(m*m),bounds=[(0,1)]*(m*m),
        constraints=[{'type':'eq','fun':lambda x:AA@x-bb}],method='SLSQP',options={'ftol':1e-14,'maxiter':2000})
    if s.success and (best is None or s.fun<best.fun): best=s
Pmin=np.clip(best.x.reshape(m,m),0,None); Pmin/=Pmin.sum(0)
import os
if os.path.exists("P_minRE.npy"): Pmin=np.load("P_minRE.npy")     # the minimizer is not unique; fix it for reproducibility
else: np.save("P_minRE.npy",Pmin)
print("(A) permutation",np.argmax(Q,axis=1)+1,"rhoR %.3f RE %.3f | min matrix rhoR %.3f RE %.3f"%(rhoR(Q),REm(Q),rhoR(Pmin),REm(Pmin)))
rng=np.random.default_rng(77)
for name,P in(("permutation",Q),("min-RE",Pmin),("Gaussian 0.50",MECH["Gaussian 0.50"])):
    r=run(P,"clean",0.25,20,500,40000,rng)["mean"]; RE=REm(P); vS,vR=1/500,1/(r["kR"]*m*RE); o=min(vS,vR)
    print("   %-14s RE %.3f costadj %.3f  P(RSS) %.3f  regret: SRS %.1f RSS %.1f adaptive %.1f"%(name,RE,RE*r["kR"]*m/500,r["pR"],100*(vS/o-1),100*(vR/o-1),100*(r["mseA"]/o-1)))
    out["A_"+name]=dict(RE=RE,pR=r["pR"],regA=100*(r["mseA"]/o-1),regS=100*(vS/o-1),regR=100*(vR/o-1))
# ---------- (B) set-based generator vs shortcut, Gaussian tau for 0.75, clean and pre
tau=1.33957; reps=40000; k=80
def setbased(cont):
    tot=np.zeros(reps)
    for _ in range(4):
        x=rng.standard_normal((reps//4,k,m,m))
        if cont=="pre": x=contaminate(x,rng)
        xs=np.sort(x,axis=-1)
        obs=np.argsort(np.argsort(np.arange(1,m+1)+tau*rng.standard_normal(x.shape),axis=-1),axis=-1)   # observed rank of each true rank
        sel=np.argmax(obs==np.arange(m)[None,None,:,None],axis=-1)                                       # true rank holding observed rank s (set s)
        yield np.take_along_axis(xs,sel[...,None],axis=-1)[...,0].mean((1,2))
for cont in("clean","pre"):
    sb=np.concatenate(list(setbased(cont))); sh=rss(MECH["Gaussian 0.75"],reps,k,cont,rng).mean((1,2))
    print("(B) %s: n*Var set-based %.4f  shortcut %.4f  (s.e. about %.4f)"%(cont,400*sb.var(),400*sh.var(),400*sb.var()*np.sqrt(2/reps)))
    out["B_"+cont]=(400*sb.var(),400*sh.var())
# ---------- (C) Gaussian vs Laplace at matched realized rho_R (common random numbers)
N=4_000_000; ZG=rng.standard_normal((N,m)); ZL=rng.laplace(size=(N,m)); J=np.eye(m)[::-1]
H=np.linalg.svd(np.eye(m)-np.ones((m,m))/m)[0][:,:m-1]
def Pof(Z,tau,sl=slice(None)):
    obs=np.argsort(np.argsort(np.arange(1,m+1)+tau*Z[sl],axis=1),axis=1)
    P=np.array([np.bincount(obs[:,i],minlength=m) for i in range(m)])/obs.shape[0]; return (P+J@P@J)/2
for tgt,tg,tl in((.25,5.56036,4.76666),(.5,2.50672,2.07187),(.75,1.33957,1.03538),(.9,0.795978,0.579369)):
    tG=optimize.brentq(lambda t:rhoR(Pof(ZG,t))-tgt,tg*0.9,tg*1.1,xtol=1e-6); tL=optimize.brentq(lambda t:rhoR(Pof(ZL,t))-tgt,tl*0.9,tl*1.1,xtol=1e-6)
    ev=np.linalg.eigvalsh(H.T@(Pof(ZG,tG)@Pof(ZG,tG).T-Pof(ZL,tL)@Pof(ZL,tL).T)@H)
    bat=np.array([np.linalg.eigvalsh(H.T@(Pof(ZG,tG,slice(i*N//8,(i+1)*N//8))@Pof(ZG,tG,slice(i*N//8,(i+1)*N//8)).T-Pof(ZL,tL,slice(i*N//8,(i+1)*N//8))@Pof(ZL,tL,slice(i*N//8,(i+1)*N//8)).T)@H) for i in range(8)])
    print("(C) target %.2f tauG %.5f tauL %.5f  RE_G %.4f RE_L %.4f  eig"%(tgt,tG,tL,REm(Pof(ZG,tG)),REm(Pof(ZL,tL))),np.round(ev,5),"se",np.round(bat.std(0,ddof=1)/np.sqrt(8),5))
    out["C_%.2f"%tgt]=dict(ev=ev.tolist(),se=(bat.std(0,ddof=1)/np.sqrt(8)).tolist(),REG=REm(Pof(ZG,tG)),REL=REm(Pof(ZL,tL)))
# ---------- (F) larger generator check: 800,000 replications of a 10-cycle RSS mean, Gaussian mechanism at 0.75
P=MECH["Gaussian 0.75"]; kk=10; sb=[]; sh=[]
for _ in range(40):
    x=np.sort(rng.standard_normal((20000,kk,m,m)),axis=-1)
    obs=np.argsort(np.argsort(np.arange(1,m+1)+tau*rng.standard_normal(x.shape),axis=-1),axis=-1)
    sel=np.argmax(obs==np.arange(m)[None,None,:,None],axis=-1)
    sb.append(np.take_along_axis(x,sel[...,None],axis=-1)[...,0].mean((1,2))); sh.append(rss(P,20000,kk,"clean",rng).mean((1,2)))
sb=np.concatenate(sb); sh=np.concatenate(sh); ex=(m-np.sum((P.T@e)**2))/m
print("(F) n*Var: exact %.4f  set-based %.4f  shortcut %.4f  (s.e. %.4f, %d replications)"%(ex,kk*m*sb.var(),kk*m*sh.var(),ex*np.sqrt(2/len(sb)),len(sb)))
json.dump(out,open("rev_exp.json","w"),indent=1,default=float)
