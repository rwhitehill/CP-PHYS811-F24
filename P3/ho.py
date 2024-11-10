import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({
    'text.latex.preamble': r'\usepackage{amsmath}',
    'text.usetex': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica']
})

from scipy.special import hermite


def V(t):
    return t**2

def get_HO_psi(u,n):
    Hn = hermite(n)
    return np.pi**(-0.25)/np.sqrt(2**n*np.math.factorial(n))*Hn(u)*np.exp(-u**2/2)


if __name__ == '__main__':
    
    x,E,psi = solve_SE(-50,50,1000,V)
    E,psi = E[:6],psi[:6]
    h   = x[1] - x[0]
    
    
    nrows,ncols = 1,1
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*ncols,5*nrows))

    img  = ax.imshow(np.log10(np.abs(psi@psi.T)),cmap='viridis')


    ticks = [0,1,2,3,4,5]
    ax.set_xticks(ticks,[r'$%d$'%_ for _ in ticks])
    ax.set_yticks(ticks,[r'$%d$'%_ for _ in ticks])
    ax.tick_params(axis='both',which='major',direction='out',labelsize=20)
    ax.set_title(r'$\displaystyle \int_{-\infty}^{\infty} {\rm d}{x} \, \tilde{\psi}^{*}_{n}(x) \tilde{\psi}_{m}(x)$',size=25,pad=30)
    ax.set_xlabel(r'$n$',size=30)
    ax.set_ylabel(r'$m$',size=30)

    cbar = fig.colorbar(img)
    cbar.ax.tick_params(labelsize=20)
    ticks = [0,-5,-10,-15]
    cbar.ax.set_yticks(ticks,[r'$1$']+[r'$10^{%d}$'%_ for _ in ticks[1:]])

    plt.show()
    fig.savefig(r'ho_orthonormality.pdf',bbox_inches='tight')
    
    nrows,ncols = 1,1
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(10*ncols,8*nrows))

    x_ = np.linspace(-5,5,1000)
    ax.plot(x_,V(x_),color='k',alpha=1)

    n    = E.shape[0]
    psi_ = np.array([get_HO_psi(x_,_) for _ in range(n)])

    scale = 1
    for i in range(n):
        sign = np.sign(psi[i][1])#/np.sign(psi_[i][1])

        ax.axhline(E[i],color='orange',ls='--',alpha=0.7)
        ax.plot(x,scale*psi[i]/np.sqrt(h) + E[i],'k.',alpha=0.8,label=r'$\tilde{\psi}_{\rm numeric}$' if i==0 else '')
        ax.plot(x_,scale*psi_[i]+E[i],'r-',lw=4,alpha=0.5,label=r'$\tilde{\psi}_{\rm exact}$' if i==0 else '')

    ax.set_xlim(-5,5)
    ax.set_ylim(0,14.5)
    ax.set_xlabel(r'$x/a$',size=30)
    ax.set_ylabel(r'$(x/a)^2$',size=30)
    ax.tick_params(axis='both',which='major',labelsize=20,direction='in')
    ax.legend(fontsize=20,loc='upper center',frameon=False)

    plt.show()
    fig.savefig(r'ho_wfs.pdf',bbox_inches='tight')