import numpy as np


def step_Euler(func,t0,q0,h):
    t1 = t0 + h
    q1 = q0 + h*func(q0,t0)
    return t1,q1,h


def step_RKF45(func,t0,q0,h,tol,adaptive=True):
    A = np.array([
        [0,0,0,0,0,0],
        [1/4,0,0,0,0,0],
        [3/32,9/32,0,0,0,0],
        [1932/2197,-7200/2197,7296/2197,0,0,0],
        [439/216,-8,3680/513,-845/4104,0,0],
        [-8/27,2,-3544/2565,1859/4104,-11/40,0]
    ])
    B = np.array([0,1/4,3/8,12/13,1,1/2])
    
    K   = np.zeros((6,q0.shape[0]))
    for i in range(6):
        K[i] = h*func(q0+np.einsum('i,ij',A[i],K),t0+B[i]*h)
    
    t1  = t0 + h
    C   = np.array([16/135,0,6656/12825,28561/56430,-9/50,2/55])
    q1  = q0 + np.einsum('i,ij',C,K)
    
    D   = np.array([1/360,0,-128/4275,-2197/75240,1/50,2/55])
    err = np.abs(max(np.einsum('i,ij',D,K) + 1e-100)) # possibly change
    
    if err < tol or not adaptive:
        if adaptive:
            h1 = min(0.9*h*(tol/err)**(1/6),2*h)
        else:
            h1 = h
        return t1,q1,h1
    else:
        h1 = min(0.9*h*(tol/err)**(1/6),h/2)
        return step_RKF45(func,t0,q0,h1,tol)
    
    
def solve_ODE(func,t0,q0,h,stop=lambda t_res,q_res: False,tol=1e-8,method='RKF45',max_iter=int(1e6),adaptive=True):
    if method == 'RKF45':
        step_func = lambda t0_,q0_,h_: step_RKF45(func,t0_,q0_,h_,tol,adaptive)
    elif method == 'modified Euler':
        step_func = lambda t0_,q0_,h_: step_mod_Euler(func,t0_,q0_,h_)
    elif method == 'Euler':
        step_func = lambda t0_,q0_,h_: step_Euler(func,t0_,q0_,h_)
    
    t_results = [t0]
    q_results = [q0]
    for i in range(max_iter):
        t1,q1,h = step_func(t_results[-1],q_results[-1],h)
        t_results.append(t1)
        q_results.append(q1)
        
        if stop(t_results,q_results):
            break
            
    return np.array(t_results),np.array(q_results)


def get_derivative(y,x):
    e1   = (y[1] - y[0])/(x[1] - x[0])
    temp = (y[2:] - y[:-2])/(x[2:] - x[:-2])
    en   = (y[-1] - y[-2])/(x[-1] - x[-2])
    temp = np.array([e1]+list(temp)+[en])
    return temp


def F_coulomb(x1,x2,Z1,Z2):
    return Z1*Z2*(x1 - x2)/np.sqrt(((x1 - x2)**2).sum())**3

