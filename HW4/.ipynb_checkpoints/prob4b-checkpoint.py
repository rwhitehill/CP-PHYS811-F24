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
    b  = 1

    a  = lambda v,t: -g - b/m*v
    t0 = 0
    v0 = 20
    
    results = [[t0,v0]]
    h   = 1e-3
    tol = 1e-4
    while results[-1][1] > 0:
        temp = step_RKF45(a,*results[-1],h,tol)
        results.append(list(temp[:-1]))
        h = temp[-1]
    results = np.array(results).T
    
    v  = interp1d(*results)
    tp = bisect(v,results[0][0],results[0][-1])
    print(tp)
    print(m/b*np.log(1+v0/(m*g/b)))
    
    nrows,ncols = 1,1
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*ncols,5*nrows))

    T = np.linspace(*results[0][[0,-1]])
    v_exact = (v0 + m*g/b)*np.exp(-b*T/m) - m*g/b
    ax.plot(T,v_exact,'k-',label=r'$\rm exact$')
    ax.plot(T,v(T),'ro',label=r'$\rm RKF45$')

    ax.legend(frameon=False,loc='upper right',fontsize=20)
    ax.set_xlabel(r'$t$',size=30)
    ax.set_ylabel(r'$v(t)$',size=30)
    ax.tick_params(axis='both',which='major',labelsize=20,direction='in')

    plt.show()
    # fig.savefig(r'prob4b.pdf',bbox_inches='tight')
