import numpy as np, itertools
from scipy import integrate, stats, optimize
from math import comb
def os_means(m):
    out=[]
    for r in range(1,m+1):
        c=r*comb(m,r)
        f=lambda x:x*c*stats.norm.cdf(x)**(r-1)*stats.norm.sf(x)**(m-r)*stats.norm.pdf(x)
        out.append(integrate.quad(f,-12,12,epsabs=1e-13)[0])
    return np.array(out)
def RE(P,e,m): return m/(m-np.sum((P.T@e)**2))
def rhoR(P,m):
    r=np.arange(1,m+1)-(m+1)/2; return r@P@r/(r@r)
# ---- m=3 counterexample
e3=os_means(3); print("e3",e3, 3/(2*np.sqrt(np.pi)))
P1=np.array([[.5,0,.5],[0,1,0],[.5,0,.5]]);P2=np.array([[.5,0,.5],[.5,0,.5],[0,1,0]])
for P in(P1,P2): print("m3 rhoR",rhoR(P,3),"rhoF",e3@P@e3/(e3@e3),"RE",RE(P,e3,3),"PTe",P.T@e3)
print(8*np.pi/(8*np.pi-9))
# ---- m=5
m=5;e=os_means(m);d=e@e/m
print("e5",e,"||e||^2",e@e,"delta",d,"REperf",1/(1-d))
rng=np.random.default_rng(20261003)
taus={("G",.25):5.56036,("L",.25):4.76666,("G",.5):2.50672,("L",.5):2.07187,("G",.75):1.33957,("L",.75):1.03538,("G",.9):0.795978,("L",.9):0.579369}
N=4_000_000
res={}
for (mech,rho),tau in taus.items():
    P=np.zeros((m,m))
    for _ in range(8):
        Z=rng.standard_normal((N//8,m)) if mech=="G" else rng.laplace(size=(N//8,m))
        V=np.arange(1,m+1)+tau*Z
        obs=np.argsort(np.argsort(V,axis=1),axis=1)  # observed rank (0-based) of true rank r
        for r in range(m): P[r]+=np.bincount(obs[:,r],minlength=m)
    P/=N
    rF=e@P@e/(e@e); q=np.sum((P.T@e)**2)/(e@e)
    resid=np.linalg.norm(P.T@e-rF*e)/np.linalg.norm(e)
    res[(mech,rho)]=(rhoR(P,m),rF,1/(1-d*rF**2),RE(P,e,m),q,resid)
    print(mech,rho,"rhoR %.4f rhoF %.4f bound %.4f RE %.4f ratio %.5f rhoF^2 %.5f resid %.2e gap %.5f"%(rhoR(P,m),rF,1/(1-d*rF**2),RE(P,e,m),q,rF**2,resid,RE(P,e,m)-1/(1-d*rF**2)))
# ---- attainable range at fixed rhoR
def rng_range(m,e,c):
    perms=[np.eye(m)[list(p)] for p in itertools.permutations(range(m))]
    rr=np.array([rhoR(Q,m) for Q in perms]); best=-1;arg=None
    for i in range(len(perms)):
        if abs(rr[i]-c)<1e-12:
            v=np.sum((perms[i].T@e)**2)
            if v>best: best=v;arg=(i,i,1.0)
        for j in range(i+1,len(perms)):
            if (rr[i]-c)*(rr[j]-c)<0:
                w=(c-rr[j])/(rr[i]-rr[j]); P=w*perms[i]+(1-w)*perms[j]
                v=np.sum((P.T@e)**2)
                if v>best: best=v;arg=(i,j,w)
    # min: convex QP
    r=np.arange(1,m+1)-(m+1)/2
    cons=[{'type':'eq','fun':lambda x:x.reshape(m,m).sum(1)-1},{'type':'eq','fun':lambda x:x.reshape(m,m).sum(0)[:-1]-1},
          {'type':'eq','fun':lambda x:r@x.reshape(m,m)@r-c*(r@r)}]
    lo=np.inf
    g=np.random.default_rng(1)
    for t in range(12):
        x0=np.full(m*m,1/m)+0.01*g.standard_normal(m*m)
        s=optimize.minimize(lambda x:np.sum((x.reshape(m,m).T@e)**2),x0,bounds=[(0,1)]*(m*m),constraints=cons,method='SLSQP',options={'ftol':1e-14,'maxiter':2000})
        if s.success: lo=min(lo,s.fun)
    return m/(m-max(lo,0)),m/(m-best),arg,perms
for c in(0,.25,.5,.75,.9):
    lo,hi,arg,perms=rng_range(5,e,c); print("m5 rhoR",c,"min RE %.4f max RE %.4f"%(lo,hi),arg[2])
lo,hi,arg,perms=rng_range(3,e3,0.0); print("m3 rhoR 0 range",lo,hi)
