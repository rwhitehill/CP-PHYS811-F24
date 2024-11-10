import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({
    'text.latex.preamble': r'\usepackage{amsmath}',
    'text.usetex': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica']
})

def V(t,b=1,N=10,V0=1):
    return V0*((t % (1+b) > 1) | (t > N*(1 + b) - b))


if __name__ == '__main__':
    
    N    = np.arange(1,9)
    spec = []
    b,V0   = 0.5,50
    for _ in N:
        x,E,psi = solve_SE(0,100,1000,lambda t: V(t,b=b,N=_,V0=V0))
        E,psi   = E[E<V0],psi[E<V0]
        spec.append(E)
        
        
    nrows,ncols = 1,1
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*ncols,5*nrows))

    x_ = np.array([1e-1,1-1e-1])
    for i in range(len(spec)):
        E = spec[i]/V0
        for _ in E:
            ax.plot(x_+i,2*[_],color='orange')

    # ax.axhline(y=1,color='k',ls='--')

    xticks = [0.5+i for i in range(len(spec))]
    ax.set_xticks(xticks,[r'$%d$'%_ for _ in N])
    ax.set_xlabel(r'$N$',size=30)
    ax.set_ylabel(r'$E/V_0$',size=30)
    ax.tick_params(axis='both',which='major',direction='in',labelsize=20)
    ax.set_ylim(0,1.2)
    ax.text(s=r'\boldmath $2 m a^2 V_0 / \hbar^2 = %d,\, b/a = %.1g$'%(V0,b),x=0.5,y=0.95,ha='center',va='top',transform=ax.transAxes,size=20)

    plt.show()
    fig.savefig(r'comb_spectrum.pdf',bbox_inches='tight')