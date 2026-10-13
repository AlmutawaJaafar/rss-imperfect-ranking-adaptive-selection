import json,numpy as np
from scipy import integrate, stats
from math import comb
m=5
d=json.load(open("cost_adaptive_results.json")); rows=d["rows"]
Pm=np.load("P_mech.npz")
# (1) thresholds with integer budgets
def osm(m,p=1):
    return np.array([integrate.quad(lambda x:x**p*r*comb(m,r)*stats.norm.cdf(x)**(r-1)*stats.norm.sf(x)**(m-r)*stats.norm.pdf(x),-12,12,epsabs=1e-13)[0] for r in range(1,m+1)])
e=osm(m); rr=np.arange(1,6)-3.0; k=e@rr/np.linalg.norm(e)/np.linalg.norm(rr); dl=e@e/m; s=np.sqrt(1-k*k)
for g,nR in((.1,450),(.25,400),(.5,330)):
    for thr in(1+g,500/nR):
        a=np.sqrt((1-1/thr)/dl); print("gamma",g,"thr %.4f"%thr,"rho* %.4f"%(a*k+np.sqrt(1-a*a)*s))
# (7) which condition gives the max always-RSS regret at gamma .1
best=[]
for r in rows:
    if r["part"]=="main" and r["gamma"]==0.1:
        for est in("mean","trimmed","huber"):
            x=r["res"][est]; o=min(x["mseS"],x["mseR"]); best.append((100*(x["mseR"]/o-1),r["mech"],r["cont"],est))
print(sorted(best)[-4:])
# (2) delta-method decision probability, clean mean
M=[osm(m,p) for p in(1,2,3,4)]           # raw moments of X_(r)
def strat(P):
    raw=[P.T@M[i] for i in range(4)]; mu=raw[0]; v=raw[1]-mu**2
    m3=raw[2]-3*mu*raw[1]+2*mu**3; m4=raw[3]-4*mu*raw[2]+6*mu**2*raw[1]-3*mu**4
    return mu,v,m3,m4
def pR_pred(P,k0,thr):
    mu,v,m3,m4=strat(P); dd=mu-mu.mean(); w=v.mean(); b=(dd**2).mean(); R=b/w
    vW=(m4-v**2).sum()/m**2; vB=4*(dd**2*v).sum()/m**2; cBW=2*(dd*m3).sum()/m**2
    s2=(vB-2*R*cBW+R*R*vW)/w**2
    return 1+R, np.sqrt(s2), stats.norm.sf(np.sqrt(k0)*(thr-1-R)/np.sqrt(s2))
names={"Gaussian 0.25":"G_0.25","Gaussian 0.50":"G_0.5","Gaussian 0.75":"G_0.75","Gaussian 0.90":"G_0.9","Laplace 0.25":"L_0.25","Laplace 0.50":"L_0.5","Laplace 0.75":"L_0.75","Laplace 0.90":"L_0.9"}
print("\ndecision probability: delta-method vs simulated")
for r in rows:
    if r["cont"]=="clean" and r["mech"] in names and r["part"] in("main","pilot","budget"):
        x=r["res"]["mean"]; thr=r["C"]/(x["kR"]*m)
        RE,sd,p=pR_pred(Pm[names[r["mech"]]],r["k0"],thr)
        print("%-6s g%.2f C%4d k0 %2d %-14s RE %.3f thr %.3f s %.3f  pred %.3f sim %.3f"%(r["part"],r["gamma"],r["C"],r["k0"],r["mech"],RE,thr,sd,p,x["pR"]))
