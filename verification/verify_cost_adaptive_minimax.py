"""Cost-aware adaptive design with the approximately minimax pilot sizes (C=500): k0 = 14, 11, 8 for gamma = 0.10, 0.25, 0.50.
Same conditions and generator as verify_cost_adaptive.py. Output: cost_adaptive_minimax.json"""
import numpy as np, json, sys, time
src=open("verify_cost_adaptive.py").read() if __import__("os").path.exists("verify_cost_adaptive.py") else open("cost_adaptive_full.py").read()
_argv=sys.argv; sys.argv=["x"]; exec(src.split('if __name__')[0]); sys.argv=_argv
REPS=int(sys.argv[1]) if len(sys.argv)>1 else 40000
rng=np.random.default_rng(20261005); out=[]; t0=time.time()
for gam,k0 in((0.10,14),(0.25,11),(0.50,8)):
    for cont in("clean","post","pre"):
        for name,P in MECH.items():
            out.append(dict(part="main",gamma=gam,cont=cont,mech=name,k0=k0,C=500,RE=float(m/(m-np.sum((P.T@e)**2))),res=run(P,cont,gam,k0,500,REPS,rng)))
    print("gamma",gam,"k0",k0,"done %.0fs"%(time.time()-t0),flush=True)
json.dump(dict(reps=REPS,rows=out),open("cost_adaptive_minimax.json","w"),indent=1)
print("total %.0fs"%(time.time()-t0))
