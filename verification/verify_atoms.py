"""Necessity of the ordering for populations with atoms: span of the score vectors e_h."""
import numpy as np
from scipy import stats
m=5
def c(u): return np.array([stats.beta.cdf(u,r,m-r+1) for r in range(1,m+1)])-u      # (B_r(u)-u)_r, lies in the sum-zero subspace
for K in range(2,8):                                     # K equally likely atoms: F takes K-1 values in (0,1)
    us=np.arange(1,K)/K; M=np.array([c(u) for u in us]).T
    print("K=%d atoms: |U|=%d  dim span{e_h} = %d  (m-1 = %d)"%(K,K-1,np.linalg.matrix_rank(M,tol=1e-10),m-1))
# direct check for a 3-atom population by brute force: e_h for random h
rng=np.random.default_rng(0); x=rng.integers(0,3,size=(400000,m)); xs=np.sort(x,axis=1)
E=[]
for _ in range(6):
    h=rng.standard_normal(3); E.append(h[xs].mean(0)-h[x].mean())
print("3 atoms, simulated e_h for 6 random h: numerical rank",np.linalg.matrix_rank(np.array(E),tol=0.02))
# counterexample pair when |U| < m-1: P1 = U, P2 = U + t w z'
K=3; us=np.arange(1,K)/K; S=np.array([c(u) for u in us]).T
Hb=np.linalg.svd(np.eye(m)-np.ones((m,m))/m)[0][:,:m-1]
w=Hb@np.linalg.svd((Hb.T@S).T)[2][-1]                    # unit vector in H orthogonal to span S
z=np.array([-2,-1,0,1,2.])/np.sqrt(10); t=0.2/np.abs(np.outer(w,z)).max()/m
P1=np.ones((m,m))/m; P2=P1+t*np.outer(w,z)
print("P2 doubly stochastic:",np.allclose(P2.sum(0),1),np.allclose(P2.sum(1),1),P2.min()>=0)
print("||P1'v||,||P2'v|| on span{e_h}:",[ (round(np.linalg.norm(P1.T@S[:,j]),12),round(np.linalg.norm(P2.T@S[:,j]),12)) for j in range(S.shape[1])])
print("eigenvalues of P1P1'-P2P2' on H:",np.round(np.linalg.eigvalsh(Hb.T@(P1@P1.T-P2@P2.T)@Hb),5))
