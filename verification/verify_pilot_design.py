"""Decision rule as an ANOVA F test; noncentral-F approximation of Pr(D=R); approximately minimax pilot size.
The rule is: choose RSS if RE_hat > 1+gamma (optionally 1+gamma_hat when the cost is misspecified)."""
import json,sys,numpy as np
from scipy import stats
def sizes(C,g,k0,m=5):
    kR=int(np.floor(C/(m*(1+g))+1e-9)); n0=k0*m; n2=int(np.floor(C-n0*(1+g)+1e-9)); return kR*m,n0,n2
def pR_F(RE,k0,thr,m=5):                  # F test: choose RSS if F > 1+(thr-1) m k0/(m-1); ncp = k0 m (RE-1)
    crit=1+(thr-1)*m*k0/(m-1); return stats.ncf.sf(crit,m-1,m*(k0-1),k0*m*(RE-1)) if RE>1+1e-12 else stats.f.sf(crit,m-1,m*(k0-1))
def regret(RE,C,g,k0,m=5,g_hat=None):
    nR,n0,n2=sizes(C,g,k0,m); NS=n0+n2; vS=1/C; vR=1/(nR*RE); vmix=(n0/RE+n2)/NS**2; p=pR_F(RE,k0,1+(g if g_hat is None else g_hat),m)
    return (p*vR+(1-p)*vmix)/min(vS,vR)-1
REPERF={3:1.9137,5:2.7702,7:3.5949}
def minimax(C,g,m=5,grid=600):
    REs=np.linspace(1,REPERF[m],grid)
    return min(((max(regret(RE,C,g,k0,m) for RE in REs),k0) for k0 in range(2,int(C/(m*(1+g)))-1)))
def worst(C,g,k0,m=5,g_hat=None,grid=600):
    return max(regret(RE,C,g,k0,m,g_hat) for RE in np.linspace(1,REPERF[m],grid))
if __name__=="__main__":
    d=json.load(open("cost_adaptive_results.json")); m=5
    err=[]
    for r in d["rows"]:
        if r["cont"]=="clean" and r["part"] in("main","pilot","budget"):
            x=r["res"]["mean"]; p=pR_F(r["RE"],r["k0"],1+r["gamma"]); err.append(abs(p-x["pR"]))
            if r["part"]=="pilot": print("k0 %2d %-14s RE %.3f  F-approx %.3f  sim %.3f"%(r["k0"],r["mech"],r["RE"],p,x["pR"]))
    print("Pr(D=R), clean mean: max abs error over %d conditions %.3f; mean %.3f"%(len(err),max(err),np.mean(err)))
    print("\npredicted worst-case regret (%), C=500, gamma=0.25, k0=5,10,20,40:"," ".join("%.1f"%(100*worst(500,.25,k)) for k in(5,10,20,40)))
    print("\napproximately minimax pilot size, normal mean")
    for mm in(3,5,7):
        for C in(250,500,1000,2000,5000) if mm==5 else (500,):
            print("m %d C %5d: "%(mm,C)+"  ".join("gamma %.2f: k0* %2d (%.1f%%)"%(g,minimax(C,g,mm)[1],100*minimax(C,g,mm)[0]) for g in(0.10,0.25,0.50)))
    print("\nmisspecified cost: worst-case predicted regret (%), C=500, m=5, minimax k0 for the true gamma")
    for g,k0 in((0.10,14),(0.25,11),(0.50,8)):
        print("true gamma %.2f k0 %d: "%(g,k0)+"  ".join("gamma_hat/gamma %.2f: %.1f"%(f,100*worst(500,g,k0,5,g*f)) for f in(0.5,0.75,1.0,1.5,2.0)))
