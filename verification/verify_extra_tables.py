"""Summaries of extra_<part>.json (set sizes, lognormal, asymmetric contamination, misspecified cost)."""
import json,sys,numpy as np,os
def mse(x,k,t): return x["q"+k]-2*t*x["m"+k]+t*t
def load(part): return json.load(open("extra_%s.json"%part)) if os.path.exists("extra_%s.json"%part) else None
def regs(d,row,est,target=None):
    x=row["res"][est]; t=d["targets"][row["cont"]][est] if target is None else target
    S,R,A=(mse(x,k,t) for k in "SRA"); o=min(S,R); return 100*(S/o-1),100*(R/o-1),100*(A/o-1),x["pR"]
for part in("setsize3","setsize7","lognormal","asym"):
    d=load(part)
    if d is None: continue
    ests=[e for e in("mean","trimmed","huber")]
    print("\n== %s (m=%d, %d replications)"%(part,d["m"],d["reps"]))
    if d["taus"]: print("   calibrated scales:",{k[4:]:round(v,4) for k,v in d["taus"].items()})
    if part in("setsize3","setsize7","lognormal"):
        print("   mean-efficiency RE(P) by mechanism:",{r["mech"]:round(r["RE"],3) for r in d["rows"] if r["gamma"]==0.25 and r["cont"]==d["rows"][0]["cont"]})
    for tgt,lab in ((None,"about the estimator's own functional"),)+(((0.0,"about the uncontaminated centre 0"),) if part=="asym" else ()):
        print("   regret (%) "+lab)
        for g in(0.10,0.25,0.50):
            R=np.array([regs(d,r,e,tgt)[:3] for r in d["rows"] if r["gamma"]==g for e in ests])
            k0=[r["k0"] for r in d["rows"] if r["gamma"]==g][0]
            line="   gamma %.2f k0 %2d (%d cond.): worst SRS %.1f RSS %.1f adaptive %.1f | mean adaptive %.1f | adaptive worst by estimator:"%(g,k0,len(R),*R.max(0),R[:,2].mean())
            print(line," ".join("%s %.1f"%(e,max(regs(d,r,e,tgt)[2] for r in d["rows"] if r["gamma"]==g)) for e in ests))
    if part=="asym":
        print("   targets:",{c:{k:round(v,4) for k,v in t.items()} for c,t in d["targets"].items()})
        r=[r for r in d["rows"] if r["gamma"]==0.25 and r["mech"]=="Gaussian 0.90" and r["cont"]=="pre"][0]
        for e in ests:
            x=r["res"][e]; print("   example (Gaussian 0.90, pre, gamma .25) %s: bias SRS %.4f RSS %.4f | variance SRS %.5f RSS %.5f | bias^2/MSE about 0 (SRS) %.2f"%(e,x["mS"],x["mR"],x["qS"]-x["mS"]**2,x["qR"]-x["mR"]**2,x["mS"]**2/x["qS"]))
    if part=="lognormal":
        print("   targets:",{k:round(v,4) for k,v in d["targets"]["clean"].items()})
        for r in d["rows"]:
            if r["gamma"]==0.25: print("   %-14s RE(mean) %.3f | "%(r["mech"],r["RE"])+" ".join("%s: eff %.3f pR %.3f regA %.1f |"%(e[:4],mse(r["res"][e],"S",d["targets"]["clean"][e])/mse(r["res"][e],"R",d["targets"]["clean"][e]),r["res"][e]["pR"],regs(d,r,e)[2]) for e in ests))
d=load("misspec")
if d:
    exec(open("verify_pilot_design.py" if os.path.exists("verify_pilot_design.py") else "pilot_design.py").read().split('if __name__')[0])
    print("\n== misspecified cost (true gamma 0.25, k0 11, clean mean): worst-case regret over 10 mechanisms, simulated vs predicted")
    for f in(0.5,0.75,1.0,1.5,2.0):
        rows=[r for r in d["rows"] if abs(r["gamma_hat"]-0.25*f)<1e-9]; v=[]
        for r in rows:
            x=r["res"]["mean"]; vS,vR=1/500,1/(x["kR"]*5*r["RE"]); v.append(100*(x["qA"]/min(vS,vR)-1))
        print("   gamma_hat/gamma %.2f: simulated worst %.1f mean %.1f | predicted worst %.1f"%(f,max(v),np.mean(v),100*worst(500,0.25,11,5,0.25*f)))
