import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({
    'text.latex.preamble': r'\usepackage{amsmath}',
    'text.usetex': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica']
})

def V(t,D=20,a=1):
    return D*(1-np.exp(-a*(t-1)))**2


if __name__ == '__main__':
    
    D,a = 20,0.75
    x,E,psi = solve_SE(0,10,1000,lambda t: V(t,D=20,a=0.75))
    E,psi   = E[E<D],psi[E<D]
    h = x[1] - x[0]
    
    nrows,ncols = 1,1
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(10*ncols,8*nrows))

    x_ = np.linspace(0,5,1000)
    ax.plot(x_,V(x_),color='k',alpha=1)

    scale = 1.8
    for i in range(len(E)):

        ax.axhline(E[i],color='orange',ls='--',alpha=0.7)
        # ax.plot(x, + E[i],'k.',alpha=0.8,label=r'$\rm numeric$' if i==0 else '')
        ax.plot(x_[::15],scale*psi[i][::15]/np.sqrt(h)+E[i],'k.',label=r'$\tilde{\psi}_{\rm numeric}$' if i==0 else '')

    ax.set_xlim(0,5)
    ax.set_ylim(0,23)
    ax.set_xlabel(r'$x/a$',size=30)
    ax.set_ylabel(r'$V(x)$',size=30)
    ax.tick_params(axis='both',which='major',labelsize=20,direction='in')
    ax.legend(fontsize=20,loc='upper right',frameon=False)

    plt.show()
    fig.savefig(r'morse_wfs.pdf',bbox_inches='tight')