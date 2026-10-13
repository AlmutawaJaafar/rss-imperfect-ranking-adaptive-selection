"""Summary of cost_adaptive_minimax.json (Table: regret with approximately minimax pilot sizes)."""
import json,numpy as np
m=5; d=json.load(open("cost_adaptive_minimax.json")); rows=d["rows"]
def reg(r,est):
    x=r["res"][est]
    if r["cont"]=="clean" and est=="mean": vS,vR=1/r["C"],1/(x["kR"]*m*r["RE"])
    else: vS,vR=x["mseS"],x["mseR"]
    o=min(vS,vR); return 100*(x["mseS"]/o-1),100*(x["mseR"]/o-1),100*(x["mseA"]/o-1)
for g in(.1,.25,.5):
    R=np.array([reg(r,e) for r in rows if r["gamma"]==g for e in("mean","trimmed","huber")])
    print("gamma %.2f k0 %d: worst case SRS %.1f RSS %.1f adaptive %.1f | mean adaptive %.1f | by estimator:"%(g,[r["k0"] for r in rows if r["gamma"]==g][0],*R.max(0),R[:,2].mean()),
          " ".join("%s %.1f"%(e,max(reg(r,e)[2] for r in rows if r["gamma"]==g)) for e in("mean","trimmed","huber")))
