import numpy as np
from scipy.integrate import quad


def trap_quad(f,a,b,N=10):
    x_i = np.linspace(a,b,N+1)
    h   = (b-a)/N
    w_i = h/2*np.array([1]+(N-1)*[2]+[1])
    f_i = f(x_i)
    return np.sum(w_i*f_i)

def rom_quad(f,a,b,N=10,R=2):
    I1  = trap_quad(f,a,b,N)
    IR  = trap_quad(f,a,b,R*N)
    err = 1/(R**2 - 1)*(IR - I1)
    return IR+err

def gauss_quad(f,a,b,n=10):
    u_i,w_i = np.polynomial.legendre.leggauss(n)
    x_i = (b-a)*u_i/2 + (a+b)/2
    f_i = f(x_i)
    Jac = (b-a)/2
    return np.sum(Jac*w_i*f_i)

def gauss_quad_improper(f,a,b,n=10):
    f_ = lambda t: f(np.tan(t))/np.cos(t)**2
    
    if b == np.inf:
        b = np.pi/2
    else:
        b = np.arctan(b)
    
    if a == -np.inf:
        a = -np.pi/2
    else:
        a = np.arctan(a)
        
    return gauss_quad(f_,a,b,n)
    
def adaptive_gauss_quad(f,a,b,n=10,reps=1e-2,aeps=1e-2):
    I1 = gauss_quad(f,a,b)
    
    m  = (a+b)/2
    I2 = gauss_quad(f,a,m,n) + gauss_quad(f,m,b,n)
    
    adiff = np.abs(I2 - I1)
    rdiff = np.abs(adiff/(I1 + 1e-100))
    if adiff < aeps or rdiff < reps:
        return I2
    else:
        return gauss_quad(f,a,m,n)+gauss_quad(f,m,b,n)
