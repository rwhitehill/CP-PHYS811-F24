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



if __name__=='__main__':
    
    setups = [
        [lambda x,t: x + np.exp(-t), 0, 0],
        [lambda x,t: x + 2*np.cos(t), 0, 1],
        [lambda x,t: t*x**2, 0, 1],
        [lambda x,t: 1.5*np.sin(2*x) - x*np.cos(t), 0, 1]
    ]
    h   = 0.1
    tf  = 1
    
    results = []
    for setup in setups:
        temp = {}
        for method in ['Euler','modified Euler','RKF45']:
            temp[method] = solve_ODE(*setup,h,tf,method=method,adaptive=False)
        results.append(temp)
        
    exact = [
        lambda t: np.sinh(t),
        lambda t: 2*np.exp(t) + np.sin(t) - np.cos(t),
        lambda t: 2/(2-t**2)
    ]

    T = np.linspace(0,1)
    exact_ = [_(T) for _ in exact]
    
    nrows,ncols=2,4
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*ncols,8),gridspec_kw={'height_ratios':[6,2]})

    color  = ['r','b','g']
    labels = [r'$\rm Euler$',r'$\rm modified~Euler$',r'${\rm RKF}45$']
    for i in range(4):
        temp = results[i]

        if i == 3:
            interp = interp1d(temp['RKF45'][0],temp['RKF45'][1],kind='cubic')

        if i != 3:
            ax[0][i].plot(T,exact_[i],'k--',label=r'$\rm exact$')

        for j,method in enumerate(temp):
            _ = temp[method]

            ax[0][i].scatter(_[0],_[1],marker='o',s=50,color=color[j],edgecolor=color[j],alpha=0.5,label=labels[j] if i==0 else '')

            if i != 3:
                r = _[1]/exact[i](_[0])
                ax[1][i].scatter(_[0],r,marker='o',s=50,color=color[j],edgecolor=color[j],alpha=0.5)
            else:
                cond = _[0] < _[0][-1]
                r = _[1][cond]/interp(_[0][cond])
                ax[1][i].scatter(_[0][cond],r,marker='o',s=50,color=color[j],edgecolor=color[j],alpha=0.5)

        ax[0][i].text(s=r'\boldmath $({\rm %s})$'%['a','b','c','d'][i],x=0.975,y=0.05,size=30,ha='right',va='bottom',transform=ax[0][i].transAxes)



        for k in range(2):
            ax[k][i].tick_params(axis='both',which='major',direction='in',labelsize=20)
            ax[k][i].set_xlim(-0.05,1.05)
        ax[1][i].set_xlabel(r'$t$',size=30)
        ax[0][i].set_xticks([],[])
        ax[1][i].set_ylim(0.83,1.03)

    ax[0][0].set_ylabel(r'$x(t)$',size=30)
    ax[1][0].set_ylabel(r'$\rm ratio$',size=30)
    ax[0][0].legend(fontsize=20,loc='upper left',frameon=False)

    fig.align_labels()
    plt.tight_layout()
    plt.show()
    # fig.savefig(r'prob2.pdf',bbox_inches='tight')