import numpy as np

def step_Euler(func,t0,x0,h):
    t1 = t0 + h
    x1 = x0 + h*func(x0,t0)
    return np.array([t1,x1,h])


def step_mod_Euler(func,t0,x0,h):
    t1,xp,h = step_Euler(func,t0,x0,h)
    x1 = x0 + 0.5*h*(func(xp,t1) + func(x0,t0))
    return np.array([t1,x1,h])


def step_RKF45(func,t0,x0,h,tol,adaptive=True):
    A = np.array([
        [0,0,0,0,0,0],
        [1/4,0,0,0,0,0],
        [3/32,9/32,0,0,0,0],
        [1932/2197,-7200/2197,7296/2197,0,0,0],
        [439/216,-8,3680/513,-845/4104,0,0],
        [-8/27,2,-3544/2565,1859/4104,-11/40,0]
    ])
    B = np.array([0,1/4,3/8,12/13,1,1/2])
    
    K = np.zeros(6)
    K[0] = h*func(x0+np.sum(A[0]*K),t0+B[0]*h)
    K[1] = h*func(x0+np.sum(A[0]*K),t0+B[1]*h)
    K[2] = h*func(x0+np.sum(A[2]*K),t0+B[2]*h)
    K[3] = h*func(x0+np.sum(A[3]*K),t0+B[3]*h)
    K[4] = h*func(x0+np.sum(A[4]*K),t0+B[4]*h)
    K[5] = h*func(x0+np.sum(A[5]*K),t0+B[5]*h)
    
    t1  = t0 + h
    x1  = x0 + np.sum(np.array([16/135,0,6656/12825,28561/56430,-9/50,2/55])*K)
    err = np.abs(np.sum(np.array([1/360,0,-128/4275,-2197/75240,1/50,2/55])*K) + 1e-100)
    
    if err < tol or not adaptive:
        if adaptive:
            h1 = min(0.9*h*(tol/err)**(1/6),2*h)
        else:
            h1 = h
        return np.array([t1,x1,h1])
    else:
        h1 = min(0.9*h*(tol/err)**(1/6),h/2)
        return step_RKF45(func,t0,x0,h1,tol)
    

def solve_ODE(func,t0,x0,h,tf,tol=1e-4,method='RKF45',max_iter=int(1e3),adaptive=True):
    if method == 'RKF45':
        step_func = lambda t0_,x0_,h_: step_RKF45(func,t0_,x0_,h_,tol,adaptive)
    elif method == 'modified Euler':
        step_func = lambda t0_,x0_,h_: step_mod_Euler(func,t0_,x0_,h_)
    elif method == 'Euler':
        step_func = lambda t0_,x0_,h_: step_Euler(func,t0_,x0_,h_)
        
    results = [[t0,x0]]
    for i in range(max_iter):
        t1,x1,h = step_func(*results[-1],h)
        results.append([t1,x1])
        
        if t1 >= tf:
            break
            
    return np.array(results).T

    