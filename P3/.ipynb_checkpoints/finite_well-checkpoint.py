import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({
    'text.latex.preamble': r'\usepackage{amsmath}',
    'text.usetex': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica']
})

from scipy.optimize import bisect

def get_finite_well_E(V0):
    f1 = lambda t: t*np.tan(t) - np.sqrt(V0 - t**2)
    f2 = lambda t: t/np.tan(t) + np.sqrt(V0 - t**2)
    
    k     = []
    ab    = np.array([1e-5,(1-1e-5)*np.pi/2])
    k_max = np.sqrt(V0)
    N     = np.ceil(2*k_max/np.pi).astype(int)
    for i in range(N):
        if ab[-1] > k_max:
            ab[-1] = (1-1e-16)*k_max
        
        f = f1 if i%2==0 else f2
        k.append(bisect(f,*ab))
        
        ab += np.pi/2
        
    E = np.array(k)**2
    return E


def get_finite_well_psi(u,V0):
    temp = get_finite_well_E(V0)
    k    = np.sqrt(temp)
    U,K  = np.meshgrid(u,k)
    
    kappa = np.sqrt(V0 - K**2)
    n     = np.arange(temp.shape[0])
    
    B = 1/np.sqrt(1 + np.sin(2*K)/2/K + np.cos(K)**2/kappa)
    B[n%2==1] = 0
    C = 1/np.sqrt(1 - np.sin(2*K)/2/K + np.sin(K)**2/kappa)
    C[n%2==0] = 0
    
    A = (B*np.cos(K) - C*np.sin(K))*np.exp(kappa)
    D = (B*np.cos(K) + C*np.sin(K))*np.exp(kappa)
    
    return temp,A*np.exp(kappa*U)*(U<-1) + (B*np.cos(K*U) + C*np.sin(K*U))*(np.abs(U) < 1) + D*np.exp(-kappa*U)*(U>1)

def V(t,V0=1):
    temp = V0*(np.abs(t) > 1).astype(float)
    return temp


if __main__ == '__main__':
    
    nrows,ncols = 1,1
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(10*ncols,8*nrows))

    v0 = 100
    N  = np.ceil(np.pi*np.sqrt(v0)/np.pi).astype(int)
    for i in range(0,N):
        k = np.linspace(1e-10+i*np.pi/2,(i+1)*np.pi/2-1e-10)
        ax.plot(k,k*np.tan(k),'C0',label=r'$k \tan{k}$' if i==0 else '')
        ax.plot(k,-k/np.tan(k),'C1',label=r'$-k \cot{k}$' if i==0 else '')

    k = np.linspace(1e-10,np.sqrt(v0)-1e-10,1000)
    ax.plot(k,np.sqrt(v0-k**2),'k--',alpha=0.7,label=r'$\sqrt{v_0 - k^2}~[v_0 = %d]$'%v0)

    E_  = get_finite_well_E(v0)
    k_  = np.sqrt(E_)
    n_  = np.arange(k_.shape[0])
    lhs = k_*(np.tan(k_)*(n_%2==0) - 1/np.tan(k_)*(n_%2==1))
    ax.scatter(k_,lhs,color='None',marker='o',edgecolor='k',s=100,lw=2,label=r'$\rm bound~states$')

    ax.set_xlim(0,1.05*np.sqrt(v0))
    ax.set_ylim(0,1.1*np.sqrt(v0))
    ax.tick_params(axis='both',which='major',direction='in',labelsize=20)
    ax.set_xlabel(r'$k$',size=30)
    ax.legend(fontsize=20,loc='lower left',frameon=True)

    plt.show()
    fig.savefig(r'finite_well_exact_solutions.pdf',bbox_inches='tight')
    
#####    

    nrows,ncols = 2,2
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*ncols,5*nrows))

    x_ = np.linspace(-10,10,1000)
    E_,psi_ = get_finite_well_psi(x_,V0)

    V0 = 32
    scale = 2
    for i in range(2):
        for j in range(2):
            xm = [[2,2],[10,10]][i][j]
            N  = [[50,200],[50,200]][i][j]

            x,E,psi = solve_SE(-xm,xm,N,lambda t: V(t,V0=V0))
            E,psi   = E[E<V0],psi[E<V0]
            h       = x[1] - x[0]

            ax_ = ax[i][j]
            ax_.plot(x_,V(x_,V0=V0),color='k',alpha=1)
            for k in range(len(E)):
                sign = np.sign(psi[k][psi[k]!=0][0])/np.sign(psi_[k][psi_[k]!=0][0])

                ax_.axhline(E[k],color='orange',ls='--',alpha=0.7)
                ax_.plot(x,scale*sign*psi[k]/np.sqrt(h) + E[k],'k.',alpha=0.8,label=r'$\tilde{\psi}_{\rm numeric}$' if k==0 else '')
                ax_.plot(x_,scale*psi_[k]+E[k],'r-',lw=4,alpha=0.5,label=r'$\tilde{\psi}_{\rm exact}$' if k==0 else '')

            ax_.set_xlim(-5,5)
            ax_.set_ylim(0,1.35*V0)
            ax_.tick_params(axis='both',which='major',labelsize=20,direction='in')

    ax[1][0].set_xlabel(r'$x/a$',size=30)
    ax[1][1].set_xlabel(r'$x/a$',size=30)
    ax[0][0].set_ylabel(r'$2 m a^2 V(x) / \hbar^2$',size=30)
    ax[1][0].set_ylabel(r'$2 m a^2 V(x) / \hbar^2$',size=30)
    ax[0][0].legend(fontsize=20,loc='upper right',frameon=False)

    fig.tight_layout()
    plt.show()
    fig.savefig(r'finite_well_wfs.pdf',bbox_inches='tight')
    
#####
    
    V0   = [100,1000,10000]
    spec = []
    for _ in V0:
        x,E,psi = solve_SE(-10,10,1000,lambda t: V(t,V0=_))
        E = E[E<_][:4]#[E<_]
        spec.append(E/E[0])

    nrows,ncols = 1,1
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*ncols,5*nrows))

    for i,_ in enumerate(spec):
        n_ = np.arange(_.shape[0]) + 1
        ax.plot(n_,_/_[0]/n_**2,ls='-.',marker='x',label=r'$v_0 = %d$'%V0[i])

    ax.axhline(y=1,color='k',lw=3,alpha=0.3)

    ax.legend(fontsize=20,loc='lower left',frameon=False)
    ax.tick_params(axis='both',which='major',direction='in',labelsize=20)
    ax.set_xlabel(r'$n$',size=30)
    ax.set_ylabel(r'$( \frac{E_n}{E_0}) / n^2$',size=30)

    plt.show()
    fig.savefig(r'finite_well_spectrum.pdf',bbox_inches='tight')