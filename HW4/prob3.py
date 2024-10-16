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

    f = lambda x,t: 1.5*np.sin(2*x) - x*np.cos(t)
    h = [1e-2,0.1,1]
    results  = []
    results1 = []
    for _ in h:
        results.append(solve_ODE(f,0,1,_,1,method='RKF45'))
        results1.append(solve_ODE(f,0,1,_,1,method='RKF45',adaptive=False))


    nrows,ncols = 1,1
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*ncols,5*nrows))

    for i,_ in enumerate(h):
        t,x = results[i]
        ax.scatter(t,x,marker='o',s=50,color='C%d'%i,edgecolor='C%d'%i,alpha=0.8,label=r'$h_0=%.1g$'%_)

        t,x = results1[i]
        ax.scatter(t,x,marker='x',s=50,color='C%d'%i,alpha=[0.3,0.8,0.8][i])


    ax.scatter([],[],marker='o',s=50,color='k',label=r'$\rm adaptive$')
    ax.scatter([],[],marker='x',s=50,color='k',label=r'$\rm static$')

    ax.tick_params(axis='both',which='major',direction='in',labelsize=20)
    ax.legend(fontsize=20,loc='upper left',frameon=False,ncol=2,columnspacing=0.5)
    ax.set_xlabel(r'$t$',size=30)
    ax.set_ylabel(r'$x(t)$',size=30)
    ax.set_xlim(-0.02,1.02)
    ax.set_ylim(0.97,1.4)

    plt.show()
    # fig.savefig(r'prob3.pdf',bbox_inches='tight')