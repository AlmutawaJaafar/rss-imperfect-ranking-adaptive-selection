"""Summaries of cost_adaptive_results.json (tables of the cost-aware adaptive section)."""
import json, numpy as np
d=json.load(open("cost_adaptive_results.json")); rows=d["rows"]; m=5
def reg(r,est):
    x=r["res"][est]
    if r["cont"]=="clean" and est=="mean": vS,vR=1/r["C"],1/(x["kR"]*m*r["RE"])      # exact fixed-strategy variances
    else: vS,vR=x["mseS"],x["mseR"]
    o=min(vS,vR); return 100*(x["mseS"]/o-1),100*(x["mseR"]/o-1),100*(x["mseA"]/o-1),x["pR"],vS,vR
print("reps",d["reps"])
print("\n== A: regret (%) over 10 mechanisms x 3 contamination x 3 estimators = 90 conditions per gamma: max / mean")
for g in(0.10,0.25,0.50):
    R=np.array([reg(r,e)[:3] for r in rows if r["part"]=="main" and r["gamma"]==g for e in("mean","trimmed","huber")])
    print(g,len(R),"max",np.round(R.max(0),1),"mean",np.round(R.mean(0),1),"min adapt %.1f"%R[:,2].min())
print("\n== A2: by estimator and contamination, max regret (SRS, RSS, adaptive)")
for g in(0.10,0.25,0.50):
    for c in("clean","post","pre"):
        for e in("mean","trimmed","huber"):
            R=np.array([reg(r,e)[:3] for r in rows if r["part"]=="main" and r["gamma"]==g and r["cont"]==c])
            print(g,c,e,np.round(R.max(0),1))
print("\n== B: clean mean, gamma=0.25, k0=20, C=500; with Theorem bound (M4=3)")
for r in rows:
    if r["part"]=="main" and r["gamma"]==0.25 and r["cont"]=="clean":
        a=reg(r,"mean"); x=r["res"]["mean"]; n0=100; nR=x["kR"]*m; NS=n0+x["n2S"]; vS,vR=a[4],a[5]; pR=a[3]; pS=1-pR
        bd=min(vR+pS/NS, 1/NS+vR*(pR+n0/nR*np.sqrt(3*pR))); o=min(vS,vR)
        print("%-14s RE %.3f eta %.3f pR %.3f  regret S %.1f R %.1f A %.1f  bound %.1f"%(r["mech"],r["RE"],r["RE"]*nR/r["C"],pR,a[0],a[1],a[2],100*(bd/o-1)))
print("\n== C: sensitivity (clean mean, gamma .25): max and mean adaptive regret over 10 mechanisms; also fixed")
for part,key,vals in(("pilot","k0",(5,10,20,40)),("budget","C",(250,1000))):
    for v in vals:
        R=np.array([reg(r,"mean")[:3] for r in rows if r["part"]==part and r[key]==v])
        print(part,v,"max",np.round(R.max(0),1),"mean",np.round(R.mean(0),1))
R=np.array([reg(r,"mean")[:3] for r in rows if r["part"]=="main" and r["gamma"]==0.25 and r["cont"]=="clean"]);print("main C=500 k0=20 max",np.round(R.max(0),1),"mean",np.round(R.mean(0),1))
print("\n== D: P(RSS) for the three estimators, gamma .25")
for c in("clean","post","pre"):
    for r in rows:
        if r["part"]=="main" and r["gamma"]==0.25 and r["cont"]==c:
            print(c,"%-14s"%r["mech"]," ".join("%s: pR %.3f effRS %.3f regA %.1f |"%(e[:4],r["res"][e]["pR"],r["res"][e]["mseS"]/r["res"][e]["mseR"],reg(r,e)[2]) for e in("mean","trimmed","huber")))
# thresholds
from scipy import integrate, stats
from math import comb
e=np.array([integrate.quad(lambda x:x*r*comb(m,r)*stats.norm.cdf(x)**(r-1)*stats.norm.sf(x)**(m-r)*stats.norm.pdf(x),-12,12)[0] for r in range(1,m+1)])
rr=np.arange(1,6)-3; k=e@rr/np.linalg.norm(e)/np.linalg.norm(rr); dl=e@e/m
for g in(.1,.25,.5):
    a=np.sqrt(g/((1+g)*dl)); print("gamma",g,"needs ||P'e||/||e|| >",round(a,4)," guaranteed if rho_R >=",round(a*k+np.sqrt((1-a*a)*(1-k*k)),4),"  max gamma for RSS ever:",round(dl/(1-dl),3))
