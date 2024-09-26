import numpy as np
from tqdm import tqdm,trange

def get_LJ_potential(P,sig=1,eps=1):
    x = P.T[0]
    y = P.T[1]
    
    X = np.tile(x,(x.shape[0],1))
    XT = X.T

    Y = np.tile(y,(y.shape[0],1))
    YT = Y.T
    
    r = np.sqrt((X - XT)**2 + (Y - YT)**2).flatten()
    r = r[r!=0]
    
    return np.sum(4*eps*((sig/r)**12 - (sig/r)**6))/2


def walk(config,h=1,freeze=None):
    step = h*(2*np.random.uniform(size=config.shape)-1)
    
    if freeze is not None:
        cond = np.tile(freeze,(2,1)).T
        step *= cond
    
    return config + step

def minimize(config,h=1,sig=1,eps=1,rtol=1e-3,atol=1e-10,max_iter=int(1e8),freeze=None,track=False):
    configs = [config]
    U = [get_LJ_potential(config)]
    
    for i in range(max_iter):
        new_config = walk(configs[-1],h,freeze)
        new_U = get_LJ_potential(new_config,sig,eps)
        
        if new_U > U[-1]:
            continue
            
        configs.append(new_config)
        U.append(new_U)

        adiff = np.abs(new_U - U[-2])
        rdiff = np.abs(adiff/U[-2])
        if adiff < atol or rdiff < rtol:
            break
            
    if track:
        return np.array(configs),np.array(U)
    else:
        return configs[-1],U[-1]

def minimize1(config,h=1,sig=1,eps=1,rtol=1e-3,atol=1e-10,max_iter=int(1e8),track=False):
    configs,U = [config],[get_LJ_potential(config)]
    N = config.shape[0]
    for i in range(N):
        freeze=np.array([j==i for j in range(N)])
        args = (h,sig,eps,rtol,atol,max_iter,freeze,True)
        configs_,U_ = minimize(configs[-1],*args)
        configs += list(configs_)
        U       += list(U_)
    
    if track:
        return np.array(configs),np.array(U)
    else:
        return configs[-1],U[-1]
