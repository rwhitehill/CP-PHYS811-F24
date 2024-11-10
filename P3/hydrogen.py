import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({
    'text.latex.preamble': r'\usepackage{amsmath}',
    'text.usetex': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica']
})

from scipy.special import assoc_laguerre


def get_H_u(x,n,l):
    p,q = 2*l+1,n-l-1
    N   = (2/n)**l*np.sqrt((2/n)**3*np.math.factorial(n-l-1)/(2*n)/np.math.factorial(n+l))
    L   = assoc_laguerre(2*x/n,q,p)
    return N*x**(l+1)*np.exp(-x/n)*L
    
def V(t,l=0):
    return l*(l+1)/t**2 - 2/t


if __name__ == '__main__':
    
    l = np.array([0,1,2])
    N = 3
    n = np.arange(1,N+1)
    E_sols,u_sols = [],[]
    for _ in l:
        x,E,u = solve_SE(0,100,1000,lambda t: V(t,_))
        E,u   = E[:N-_],u[:N-_]
        h = x[1] - x[0]
        u = u/np.sqrt(h)
        E_sols.append(_*[None]+list(E))
        u_sols.append(_*[None]+list(u))

    E_sols,u_sols = E_sols[::-1],u_sols[::-1]
    
    
    
    nrows,ncols = 3,3
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*ncols,5*nrows))

    L,temp = np.meshgrid(l,n)
    L,temp = L.T,temp.T

    xmax = np.array([10,20,40])
    ymax = np.array([0.35,0.45,0.75])
    skip = np.array([2,3,5])

    for i in range(nrows):
        for j in range(ncols):

            ax_ = ax[i][j]

            if E_sols[i][j] is not None:

                x_ = np.linspace(0,xmax[j],1000)
                u_ = get_H_u(x_,j+1,ncols-i-1)
                ax_.plot(x_,u_,'r-',lw=5,alpha=0.5,label=r'$\psi_{\rm exact}(u)/\sqrt{a}$' if i==0 else '')

                u    = u_sols[i][j]
                sign = np.sign(u[1])/np.sign(u_[1])
                ax_.plot(x[::skip[j]],sign*u[::skip[j]],'k.',alpha=0.8,label=r'$\psi_{\rm numeric}(u)$' if i==0 else '')

                ax_.set_xlim(1e-2,xmax[j])
                ax_.set_ylim(-ymax[i],ymax[i])

                ax_.text(s=r'\boldmath $u_{%d%d}(\frac{r}{a})$'%(j+1,nrows-i-1),size=30,x=0.8,y=0.9,ha='center',va='top',transform=ax_.transAxes,
                         bbox=dict(facecolor='none',edgecolor='black',lw=3,boxstyle='round,pad=0.5'))
                ax_.text(s=r'$\varepsilon_{%d%d} = %.3g$'%(j+1,nrows-i-1,E_sols[i][j]),size=30,x=0.8,y=0.7,ha='center',va='top',transform=ax_.transAxes)

            else:
                ax_.spines['bottom'].set_visible(False)
                ax_.spines['left'].set_visible(False)
                ax_.set_yticks([],[])
                ax_.set_xticks([],[])

            ax_.spines['top'].set_visible(False)
            ax_.spines['right'].set_visible(False)
            ax_.spines['bottom'].set_position('zero')
            ax_.tick_params(axis='both',which='major',direction='in',labelsize=20)

    # ax[0][0].plot([],[],'r-',lw=5,alpha=0.5,label=r'$\tilde{\psi}_{\rm exact}$')
    # ax[0][0].plot([],[],'k.',alpha=0.8,label=r'$\tilde{\psi}_{\rm numeric}$')
    # ax[0][0].legend(fontsize=30,loc='center',frameon=False)

    fig.tight_layout(h_pad=3)
    plt.show()
    fig.savefig(r'hydrogen_wfs.pdf',bbox_inches='tight')