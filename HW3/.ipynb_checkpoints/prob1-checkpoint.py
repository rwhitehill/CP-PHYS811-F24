import numpy as np
from scipy.integrate import quad
from main import *

import matplotlib.pyplot as plt
plt.rcParams.update({
    'text.latex.preamble': r'\usepackage{amsmath}',
    'text.usetex': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica']
})

if __name__ == '__main__':
    
    sets = {
        1: [lambda t: t**3,[0,1],0.25],
        2: [lambda t: np.exp(t),[0,1],np.e-1],
        3: [lambda t: np.sin(t),[0,np.pi],2]
    }
    
    N = 10*np.arange(1,11)
    results = {1:{},2:{},3:{}}
    for i in sets:
        f,ab,exact = sets[i]
        results[i]['trap']    = np.array([trap_quad(f,*ab,_) for _ in N])
        results[i]['rom']     = np.array([rom_quad(f,*ab,int(_/2),2) for _ in N])
        results[i]['gauss']   = gauss_quad(f,*ab)
        results[i]['scipy_v'] = quad(f,*ab)[0]
        results[i]['scipy_e'] = quad(f,*ab)[1]
        results[i]['exact']   = exact
    
    nrows,ncols = 2,3
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*ncols,7),gridspec_kw={'height_ratios': [6,1]})

    for i in results:
        ax[0][i-1].axhline(y=results[i]['exact'],color='k',ls='--',label=r'$\rm exact$')
        ax[0][i-1].fill_between(N,results[i]['scipy_v']+results[i]['scipy_e'],results[i]['scipy_v']-results[i]['scipy_e'],color='g',label=r'$\rm scipy~quad$')
        ax[0][i-1].axhline(y=results[i]['gauss'],color='k',alpha=0.3,lw=4,label=r'${\rm Gauss}~(n=10)$')
        ax[0][i-1].plot(N,results[i]['trap'],color='r',ls='--',marker='.',markersize=10,label=r'$\rm trapezoid$')
        ax[0][i-1].plot(N,results[i]['rom'],color='c',ls='--',marker='x',markersize=10,label=r'$\rm Romberg$')

        ax[1][i-1].axhline(y=1,color='k',ls='--',label=r'$\rm exact$')
        ax[1][i-1].fill_between(N,(results[i]['scipy_v']+results[i]['scipy_e'])/results[i]['exact'],(results[i]['scipy_v']-results[i]['scipy_e'])/results[i]['exact'],color='g')
        ax[1][i-1].axhline(y=results[i]['gauss']/results[i]['exact'],color='k',alpha=0.3,lw=4)
        ax[1][i-1].plot(N,results[i]['trap']/results[i]['exact'],color='r',ls='--',marker='.',markersize=10)
        ax[1][i-1].plot(N,results[i]['rom']/results[i]['exact'],color='c',ls='--',marker='x',markersize=10)

        ax[1][i-1].set_xlabel(r'$N$',size=30)
        ax[1][i-1].set_ylim(0.97,1.03)
        for j in range(2):
            ax[j][i-1].tick_params(axis='both',which='major',direction='in',labelsize=20)
        ax[0][i-1].tick_params(axis='x',which='major',direction='in',labelsize=0)

    ax[0][0].legend(fontsize=20,loc='upper right',frameon=False)    

    fig.tight_layout(h_pad=0,w_pad=2)
    plt.show()
    fig.savefig(r'prob1.pdf',bbox_inches='tight')
    