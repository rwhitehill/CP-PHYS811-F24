import numpy as np
from scipy.linalg import eigh_tridiagonal

def solve_SE(x0,x1,n,V):
    x = np.linspace(x0,x1,n)
    h = x[1] - x[0]
    
    d = -np.full(n-2,-2)/h**2 + V(x[1:-1])
    e = -np.full(n-3,1)/h**2
    
    E,psi = eigh_tridiagonal(d,e)
    psi   = np.vstack(([np.zeros(E.shape[0]),psi,np.zeros(E.shape[0])]))
    return x,E,psi.T