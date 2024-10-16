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
    
    k  = 0.05
    TM = 90

    f  = lambda T,t: -k*(T - TM)

    T0 = 20
    tf = 10
    h  = tf/100
    results1 = solve_ODE(f,0,T0,h,tf,method='RKF45',adaptive=True,tol=1e-4)
    results2 = solve_ODE(f,0,T0,h,tf,method='RKF45',adaptive=True,tol=1e-6)
    
    T = interp1d(*results1)
    print(T(10))

    T = interp1d(*results2)
    print(T(10))

    print(TM + (T0 - TM)*np.exp(-k*tf))
    
    
    nrows,ncols = 1,1
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*ncols,5*nrows))

    t = np.linspace(0,10)
    ax.plot(t,TM + (T0 - TM)*np.exp(-k*t),'k',label=r'$\rm exact$')
    ax.plot(results1[0],results1[1],'ro',alpha=0.7,label=r'$\epsilon=10^{-4}$')
    ax.plot(results2[0],results2[1],'co',alpha=0.7,label=r'$\epsilon=10^{-6}$')

    ax.legend(fontsize=20,frameon=False,loc='upper left')
    ax.set_xlabel(r'$t$',size=30)
    ax.set_ylabel(r'$T(t)$',size=30)
    ax.tick_params(axis='both',which='major',direction='in',labelsize=20)
    ax.set_xlim(-0.5,10.5)

    plt.show()
    # fig.savefig(r'prob4a.pdf',bbox_inches='tight')

    