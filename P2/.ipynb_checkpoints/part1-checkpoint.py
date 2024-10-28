import numpy as np
from main import *

import matplotlib.pyplot as plt
plt.rcParams.update({
    'text.latex.preamble': r'\usepackage{amsmath}',
    'text.usetex': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica']
})


if __name__ == '__main__':

    def f(q,t,m1,Z1,Z2):
        F_x,F_y = F_coulomb(q[[0,1]],np.array([0,0]),Z1,Z2)
        return np.array([q[2],q[3],F_x/m1,F_y/m1])

    x0 = -50
    b  = 3
    v0 = 1
    q0 = np.array([x0,b,v0,0])
    Z1,Z2,m1 = 1,1,1

    t0 = 0
    h  = 1e-3

    def stop(t_res,q_res,atol=1e-6,rtol=1e-6):
        th_p  = np.arctan(q_res[-2][3]/q_res[-2][2])
        th_c  = np.arctan(q_res[-1][3]/q_res[-1][2])
        cond1 = np.abs(th_p - th_c) < atol or np.abs(1 - th_c/(th_p + 1e-100)) < rtol
        cond2 = np.sum(q_res[-1][:2]**2) - np.sum(q_res[0][:2]**2) > 0

        return cond1 and cond2

    t,q = solve_ODE(lambda q_,t_: f(q_,t_,m1,Z1,Z2),t0,q0,h,stop,method='RKF45',tol=1e-6,adaptive=False)    
    x,y,vx,vy = q.T

    E = 0.5*(vx**2 + vy**2) + 1/np.sqrt(x**2 + y**2)
    L = x*vy + y*vx

    nrows,ncols = 1,3
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*ncols,5*nrows))

    ax[0].plot(x,y)
    ax[0].plot(x[0],y[0],color='k',marker='o')
    ax[0].plot([0],[0],color='k',marker='o')
    ax[0].set_xlabel(r'$x$',size=30)
    ax[0].set_ylabel(r'$y$',size=30)
    ax[0].tick_params(axis='both',which='major',direction='in',labelsize=20)

    ax[1].plot(t,E)
    ax[1].set_ylim(E[0]-1e-2,E[1]+1e-2)
    ax[1].set_xlabel(r'$t$',size=30)
    ax[1].set_ylabel(r'$E$',size=30)
    ax[1].tick_params(axis='both',which='major',direction='in',labelsize=20)

    ax[2].plot(t,L)
    ax[2].set_xlabel(r'$t$',size=30)
    ax[2].set_ylabel(r'$L_z$',size=30)
    ax[2].tick_params(axis='both',which='major',direction='in',labelsize=20)

    fig.tight_layout()
    plt.show()
    # fig.savefig(r'test_solver.pdf',bbox_inches='tight')


    b = np.logspace(-3,2,50)
    q_results = np.zeros((b.shape[0],4))
    for i in range(b.shape[0]):
        q0 = np.array([x0,b[i],v0,0])
        t,q = solve_ODE(lambda q_,t_: f(q_,t_,m1=1,Z1=1,Z2=2),t0,q0,h,stop,method='RKF45',tol=1e-16,adaptive=True)
        q_results[i] = q[-1]

        x,y,vx,vy = q.T

    theta = np.arctan2(q_results[:,3],q_results[:,2])
    dsig = b**2*np.log(10)/np.sin(theta)/np.abs(get_derivative(theta,np.log10(b)))


    nrows,ncols = 1,2
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*ncols,5*nrows))

    Z1,Z2,m1 = 1,2,1

    exact = 2*np.arctan(Z1*Z2/b/m1/v0**2)
    ax[0].plot(b,theta,ls='None',marker='o',color='c',alpha=0.6,label=r'$\rm numeric~approximation$')
    ax[0].plot(b,exact,'k-',alpha=0.5,label=r'$\rm analytic~solution$')
    ax[0].legend(fontsize=20,loc='lower left',frameon=False)

    ax[0].set_xlabel(r'$b$',size=30)
    ax[0].text(s=r'\boldmath $\theta$',x=0.95,y=0.95,size=30,ha='right',va='top',transform=ax[0].transAxes)
    ax[0].tick_params(axis='both',which='major',direction='in',labelsize=20,pad=7)
    ax[0].semilogx()

    yticks  = [0,np.pi/4,np.pi/2,3*np.pi/4,np.pi]
    ylabels = [r'$0$',r'$\pi/4$',r'$\pi/2$',r'$3\pi/4$',r'$\pi$']
    ax[0].set_yticks(yticks,ylabels)


    exact = (Z1*Z2/4/(m1*v0**2/2)/np.sin(theta/2)**2)**2
    ax[1].plot(theta,dsig,ls='None',marker='o',color='c',alpha=0.5)
    ax[1].plot(theta,exact,'k-')

    ax[1].set_xlabel(r'$\theta$',size=30)
    ax[1].text(s=r'\boldmath $\displaystyle \frac{{\rm d} \sigma}{{\rm d} \Omega}$',x=0.95,y=0.95,size=30,ha='right',va='top',transform=ax[1].transAxes)
    ax[1].tick_params(axis='both',which='major',direction='in',labelsize=20)
    ax[1].tick_params(axis='both',which='minor',direction='in')
    ax[1].semilogy()

    xticks  = [0,np.pi/4,np.pi/2,3*np.pi/4,np.pi]
    xlabels = [r'$0$',r'$\pi/4$',r'$\pi/2$',r'$3\pi/4$',r'$\pi$'] 
    ax[1].set_xticks(xticks,xlabels)

    fig.tight_layout()
    plt.show()
    # fig.savefig(r'part1-xsec.pdf',bbox_inches='tight')