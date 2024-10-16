import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import bisect
from main import *

import matplotlib.pyplot as plt
plt.rcParams.update({
    'text.latex.preamble': r'\usepackage{amsmath}',
    'text.usetex': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica']
})

if __name__ == '__main__':
    
    m  = 1
    g  = 9.80665
    c  = 0.1

    a  = lambda v,t: -g - c/m*v**2
    t0 = 0
    v0 = 20
    
    results = [[t0,v0]]
    h   = 1e-3
    tol = 1e-6
    while results[-1][1] > 0:
        temp = step_RKF45(a,*results[-1],h,tol)
        results.append(list(temp[:-1]))
        h = temp[-1]
    results = np.array(results).T
    
    v  = interp1d(*results)
    tp = bisect(v,results[0][-2],results[0][-1])
    print(tp)

    vt  = np.sqrt(m*g/c)
    tau = np.sqrt(m/c/g)
    print(tau*np.arctan(v0/vt))
    
    nrows,ncols = 1,1
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*ncols,5*nrows))

    T = np.linspace(*results[0][[0,-1]])

    v_exact = vt*np.tan(np.arctan(v0/vt) - T/tau)
    ax.plot(T,v_exact,'k-',label=r'$\rm exact$')
    ax.plot(T,v(T),'ro',label=r'$\rm RKF45$')
    ax.set_xlabel(r'$t$',size=30)
    ax.set_ylabel(r'$v(t)$',size=30)
    ax.tick_params(axis='both',which='major',direction='in',labelsize=20)
    ax.legend(fontsize=20,loc='upper right',frameon=False)

    plt.show()
    # fig.savefig('prob4c.pdf',bbox_inches='tight')
    