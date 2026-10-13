"""Numerical checks for the universal-ordering theorem, the spectral bounds and the
closed-form rho_R efficiency guarantee (standard normal, m = 5)."""
import numpy as np, itertools
from scipy import integrate, stats, optimize
from math import comb
m=5
def os_means(m):
    out=[]
    for r in range(1,m+1):
        c=r*comb(m,r)
        f=lambda x:x*c*stats.norm.cdf(x)**(r-1)*stats.norm.sf(x)**(m-r)*stats.norm.pdf(x)
        out.append(integrate.quad(f,-12,12,epsabs=1e-13)[0])
    return np.array(out)
e=os_means(m); r=np.arange(1,m+1)-(m+1)/2; d=e@e/m
kap=e@r/np.linalg.norm(e)/np.linalg.norm(r)
print("kappa = %.6f  delta = %.5f"%(kap,d))
H=np.linalg.svd(np.eye(m)-np.ones((m,m))/m)[0][:,:m-1]      # orthonormal basis of the sum-zero subspace
def spec(P): return np.linalg.eigvalsh(H.T@P@P.T@H)
def rhoR(P): return r@P@r/(r@r)
def REm(P): return m/(m-np.sum((P.T@e)**2))
def closed(c): 
    a=kap*c-np.sqrt((1-kap**2)*(1-c**2)); a=max(a,0); return 1/(1-d*a*a)
# ---------- (A) beta densities: surjectivity and exact check of the variance identity for a general h
def b(rr,u): return rr*comb(m,rr)*u**(rr-1)*(1-u)**(m-rr)
G=np.array([[integrate.quad(lambda u:b(i,u)*b(j,u),0,1)[0] for j in range(1,m+1)] for i in range(1,m+1)])
print("Gram matrix of beta densities: smallest eigenvalue %.4f (nonsingular)"%np.linalg.eigvalsh(G)[0])
def h_for(v):                       # bounded h~ on (0,1) with e_h = v
    c=np.linalg.solve(G,v); return lambda u:sum(c[i]*b(i+1,u) for i in range(m))
def exact_var_sum(P,ht):            # sum_s Var_{g_s} h, and m*sigma_h^2, by quadrature on the uniform scale
    m1=np.array([integrate.quad(lambda u:ht(u)*b(i,u),0,1)[0] for i in range(1,m+1)])
    m2=np.array([integrate.quad(lambda u:ht(u)**2*b(i,u),0,1)[0] for i in range(1,m+1)])
    mu=P.T@m1; return np.sum(P.T@m2-mu**2), m*(m2.mean()-m1.mean()**2), m1-m1.mean()
g=np.random.default_rng(7)
def rand_ds():
    w=g.dirichlet(np.ones(6)); perms=[np.eye(m)[g.permutation(m)] for _ in range(6)]
    return sum(a*Q for a,Q in zip(w,perms))
mx=0
for _ in range(20):
    P=rand_ds(); v=H@g.standard_normal(m-1); ht=h_for(v)
    vs,ms,eh=exact_var_sum(P,ht); mx=max(mx,abs(ms-vs-np.sum((P.T@eh)**2)),np.abs(eh-v).max())
print("identity m*sigma_h^2 - sum Var_gs(h) = ||P^T e_h||^2 and e_h = v: max abs error %.1e"%mx)
# ---------- (B) non-comparable pair: explicit targets ordered in opposite directions
P1=np.array([[.5,0,.5],[0,1,0],[.5,0,.5]]);P2=np.array([[.5,0,.5],[.5,0,.5],[0,1,0]])
H3=np.linalg.svd(np.eye(3)-np.ones((3,3))/3)[0][:,:2]
print("m=3 example: eigenvalues of P1P1'-P2P2' on H:",np.round(np.linalg.eigvalsh(H3.T@(P1@P1.T-P2@P2.T)@H3),4))
# garbling: P Q is dominated by P
worst=min(np.linalg.eigvalsh(H.T@(P@P.T-(P@Q)@(P@Q).T)@H)[0] for P,Q in [(rand_ds(),rand_ds()) for _ in range(2000)])
print("garbling P vs PQ, 2000 random pairs: smallest eigenvalue of difference %.2e (>= 0 up to rounding)"%worst)
# ---------- (C) closed-form guarantee vs exact minimum, and vs random matrices
A=[];bb=[]
for i in range(m):
    a=np.zeros((m,m));a[i,:]=1;A.append(a.ravel());bb.append(1)
for j in range(m-1):
    a=np.zeros((m,m));a[:,j]=1;A.append(a.ravel());bb.append(1)
for c in(.25,.5,.75,.9):
    AA=np.array(A+[np.outer(r,r).ravel()]);b2=np.array(bb+[c*(r@r)])
    best=np.inf
    for t in range(8):
        x0=np.full(m*m,1/m)+0.01*g.standard_normal(m*m)
        s=optimize.minimize(lambda x:np.sum((x.reshape(m,m).T@e)**2),x0,bounds=[(0,1)]*(m*m),method='SLSQP',
            constraints=[{'type':'eq','fun':lambda x:AA@x-b2}],options={'ftol':1e-14,'maxiter':2000})
        if s.success: best=min(best,s.fun)
    print("rho_R=%.2f  closed-form bound %.4f   exact minimum %.4f"%(c,closed(c),m/(m-best)))
viol=0
for _ in range(200000):
    P=rand_ds(); viol+= REm(P) < closed(rhoR(P))-1e-12
print("closed-form bound violated in %d of 200000 random doubly stochastic matrices"%viol)
# ---------- (D) calibrated mechanisms: spectrum of P P^T on H, and pairwise comparison
taus={("G",.25):5.56036,("L",.25):4.76666,("G",.5):2.50672,("L",.5):2.07187,("G",.75):1.33957,("L",.75):1.03538,("G",.9):0.795978,("L",.9):0.579369}
rng=np.random.default_rng(20261003); NB=8; N=2_000_000; J=np.eye(m)[::-1]
Pb={}
for (mech,rho),tau in taus.items():
    Ps=[]
    for _ in range(NB):
        Z=rng.standard_normal((N,m)) if mech=="G" else rng.laplace(size=(N,m))
        obs=np.argsort(np.argsort(np.arange(1,m+1)+tau*Z,axis=1),axis=1)
        P=np.array([np.bincount(obs[:,i],minlength=m) for i in range(m)])/N
        Ps.append((P+J@P@J)/2)                      # reflection symmetry of the mechanism
    Pb[(mech,rho)]=Ps
def stat(f,*keys):
    full=f(*[np.mean(Pb[k],axis=0) for k in keys]); bat=np.array([f(*[Pb[k][i] for k in keys]) for i in range(NB)])
    return full, bat.std(axis=0,ddof=1)/np.sqrt(NB)
print("\nspectrum of P P^T on H (lambda_min, lambda_max), with batch standard errors; target-free bounds for the mean")
for k in taus:
    ev,se=stat(lambda P:spec(P)[[0,-1]],k); P=np.mean(Pb[k],axis=0)
    print(k,"lam_min %.4f (%.4f) lam_max %.4f (%.4f)  RE %.4f in [%.4f, %.4f]"%(ev[0],se[0],ev[1],se[1],REm(P),1/(1-d*ev[0]),1/(1-d*ev[1])))
print("\nGaussian vs Laplace at the same target: eigenvalues of P_G P_G' - P_L P_L' on H (batch s.e.)")
for rho in(.25,.5,.75,.9):
    ev,se=stat(lambda A_,B_:np.linalg.eigvalsh(H.T@(A_@A_.T-B_@B_.T)@H),("G",rho),("L",rho))
    print(rho,np.round(ev,5),"se",np.round(se,5))
print("\nwithin mechanism, higher target vs lower target: smallest eigenvalue of difference (batch s.e.)")
for mech in "GL":
    for hi,lo in((.5,.25),(.75,.5),(.9,.75)):
        ev,se=stat(lambda A_,B_:np.linalg.eigvalsh(H.T@(A_@A_.T-B_@B_.T)@H)[:1],(mech,hi),(mech,lo))
        print(mech,hi,"vs",lo,"%.4f (%.4f)"%(ev[0],se[0]))
np.savez("P_mech.npz",**{"%s_%s"%k:np.mean(v,axis=0) for k,v in Pb.items()})
