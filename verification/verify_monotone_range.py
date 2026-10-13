"""Attainable range of RE at fixed rho_R over stochastically monotone doubly stochastic matrices (normal, m=5)."""
import numpy as np, sys, time
from scipy import optimize, integrate, stats
from scipy.optimize import linprog
from math import comb
m=int(sys.argv[1]) if len(sys.argv)>1 else 5
e=np.array([integrate.quad(lambda x:x*r*comb(m,r)*stats.norm.cdf(x)**(r-1)*stats.norm.sf(x)**(m-r)*stats.norm.pdf(x),-12,12,epsabs=1e-13)[0] for r in range(1,m+1)])
rr=np.arange(1,m+1)-(m+1)/2; dl=e@e/m; J=np.eye(m)[::-1]
idx=lambda r,s:r*m+s
Aeq=[];beq=[]
for i in range(m):
    a=np.zeros(m*m);a[[idx(i,s) for s in range(m)]]=1;Aeq.append(a);beq.append(1)
for j in range(m-1):
    a=np.zeros(m*m);a[[idx(r,j) for r in range(m)]]=1;Aeq.append(a);beq.append(1)
Aub=[]                                   # monotone: sum_{s<=j} P[r+1,s] - P[r,s] <= 0
for r in range(m-1):
    for j in range(m-1):
        a=np.zeros(m*m)
        for s in range(j+1): a[idx(r+1,s)]+=1; a[idx(r,s)]-=1
        Aub.append(a)
Aub=np.array(Aub); bub=np.zeros(len(Aub))
rho_row=np.outer(rr,rr).ravel()/(rr@rr)
def cons(c,sym):
    A=Aeq+[rho_row]; b=beq+[c]
    if sym:
        for r in range(m):
            for s in range(m):
                if (r,s)<(m-1-r,m-1-s):
                    a=np.zeros(m*m);a[idx(r,s)]=1;a[idx(m-1-r,m-1-s)]=-1;A.append(a);b.append(0)
    return np.array(A),np.array(b)
def lp(c,u,sym):                         # max e' P u
    A,b=cons(c,sym); res=linprog(-np.outer(e,u).ravel(),A_ub=Aub,b_ub=bub,A_eq=A,b_eq=b,bounds=(0,1),method="highs")
    return (-res.fun,res.x.reshape(m,m)) if res.status==0 else (None,None)
def REq(q): return 1/(1-dl*q/(e@e))
def minq(c):
    A,b=cons(c,False); best=np.inf; g=np.random.default_rng(0)
    x0=(c*np.eye(m)+(1-c)*np.ones((m,m))/m).ravel()
    for t in range(6):
        s=optimize.minimize(lambda x:np.sum((x.reshape(m,m).T@e)**2),x0+ (0 if t==0 else 0.005*g.standard_normal(m*m)),jac=lambda x:(2*np.outer(e,x.reshape(m,m).T@e)).ravel(),bounds=[(0,1)]*(m*m),
            constraints=[{'type':'eq','fun':lambda x:A@x-b,'jac':lambda x:A},{'type':'ineq','fun':lambda x:-Aub@x,'jac':lambda x:-Aub}],method='SLSQP',options={'ftol':1e-15,'maxiter':3000})
        if s.success and np.abs(A@s.x-b).max()<1e-8 and (Aub@s.x).max()<1e-8: best=min(best,s.fun)
    return best
def max_alt(c,sym,starts,seed=0):
    g=np.random.default_rng(seed); best=0;Pb=None
    for t in range(starts):
        u=g.standard_normal(m); u-=u.mean()
        if sym: u=(u-u[::-1])/2
        u/=np.linalg.norm(u)
        for it in range(60):
            v,P=lp(c,u,sym); w=P.T@e; nw=np.linalg.norm(w)
            if nw<1e-12 or np.linalg.norm(w/nw-u)<1e-10: break
            u=w/nw
        q=np.sum((P.T@e)**2)
        if q>best: best=q;Pb=P
    return best,Pb
def max_sym_cert(c,n=8000):              # antisymmetric directions lie on a circle when m=4,5: certified by a grid + Lipschitz bound
    b1=np.zeros(m);b1[0]=-1;b1[-1]=1;b1/=np.linalg.norm(b1); b2=np.zeros(m);b2[1]=-1;b2[-2]=1;b2/=np.linalg.norm(b2)
    th=np.linspace(0,2*np.pi,n,endpoint=False); vals=np.array([lp(c,np.cos(t)*b1+np.sin(t)*b2,True)[0] for t in th])
    return vals.max()**2, (vals.max()+np.linalg.norm(e)*(2*np.pi/n)/2)**2     # value on the grid, certified upper bound
if __name__=="__main__":
    print("m=%d  RE_perf %.4f"%(m,1/(1-dl)))
    for c in(0.25,0.5,0.75,0.9):
        t0=time.time(); qmin=minq(c); qs,_=max_alt(c,True,100); g,ub=max_sym_cert(c)
        f1,_=max_alt(c,False,1500,seed=1); f2,_=max_alt(c,False,1500,seed=2)
        print("rho_R %.2f  min RE %.4f | symmetric max: multi-start %.4f, grid %.4f, certified upper bound %.4f | all monotone (multi-start): seed 1 %.4f, seed 2 %.4f  (%.0fs)"%(c,REq(qmin),REq(qs),REq(g),REq(ub),REq(f1),REq(f2),time.time()-t0),flush=True)
