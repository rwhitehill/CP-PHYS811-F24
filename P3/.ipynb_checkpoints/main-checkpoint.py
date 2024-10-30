import numpy as np
import scipy

def get_H(x0,x1,n,V):
    x = np.linspace(x0,x1,n)
    h = x[1] - x[0]
    
    T_ = -(np.diag(np.full(n-3,1),k=-1) + np.diag(np.full(n-2,-2),k=0) + np.diag(np.full(n-3,1),k=1))/2/h**2
    V_ = np.diag(V(x[1:-1]))
    H  = T_ + V_
    
    return x,H




