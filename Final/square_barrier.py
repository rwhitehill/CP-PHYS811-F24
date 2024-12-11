import sys,os
import multiprocessing
from joblib import Parallel, delayed
import time

import numpy as np
import sympy as sp

import matplotlib.pyplot as plt
plt.rcParams.update({
    'text.latex.preamble': r'\usepackage{amsmath}',
    'text.usetex': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica']
})
from matplotlib import cm,colors


def U_psi_ft(dt,x,psi0,m=1,V=lambda t: np.full(t.size,0)):
    dx = x[1] - x[0]
    N  = x.size
    dk = 2*np.pi/N/dx
    
    k  = np.fft.fftfreq(N)
    k *= dk/(k[1] - k[0])
    
    psi_t  = np.fft.fft(psi0,norm='ortho')
    # psi_t *= dx*np.sqrt(N)/np.sqrt(2*np.pi)*np.exp(-complex(0,1)*k*x[0])
    psi_t *= np.exp(-complex(0,1)*k**2*dt/2/m)
    
    return np.exp(-complex(0,1)*V(x)*dt)*np.fft.ifft(psi_t,norm='ortho')


def Psi0_gwp(x,m=1,sig=1,k=0,x0=0):
    return (2*np.pi*sig**2)**(-1/4)*np.exp(-(x-x0)**2/4/sig**2 + complex(0,1)*k*x)
    
def Psi2_gwp(x,t,m=1,sig=1,k=0,x0=0):
    return 1/np.sqrt(2*np.pi*sig**2)*2*sig**2*m/np.sqrt(4*sig**4*m**2+t**2)*np.exp(-2*m**2*sig**2*(x-x0-k*t/m)**2/(4*sig**4*m**2+t**2))


def V(x,a=1,V0=1):
    return V0*(np.abs(x)<a/2)


def A2(k,m=1,V0=1,a=1):
    if k**2 < 2*m*V0:
        kappa = np.sqrt(2*m*V0-k**2)
        return 4*m**2*V0**2*np.sinh(kappa*a)**2/(4*kappa**2*k**2 + 4*m**2*V0**2*np.sinh(kappa*a)**2)
    elif k**2 == 2*m*V0:
        return k**2*a**2/(4 + a**2*k**2)
    else:
        kp = np.sqrt(k**2 - 2*m*V0)
        return 4*m**2*V0**2*np.sin(kp*a)**2/(4*kp**2*k**2 + 4*m**2*V0**2*np.sin(kp*a)**2)
    
def B2(k,m=1,V0=1,a=1):
    return 1 - A2(k,m,V0,a)


def get_RT(k,m=1,M1=10,M2=500,dt=0.001,V0=1,a=1,max_iter=int(1e6),atol=1e-5,rtol=1e-3):
    sig_x = 5*M1/k
    x0    = -(a/2 + 5*sig_x)
    xm    = 10*sig_x + a/2
    dx    = sig_x/M2
    N     = np.round(2*xm/dx).astype(int)
    # print(-2*x0*m/k/dt)
    
    x  = np.linspace(-xm,xm,N)
    dx = x[1] - x[0]
    li = x < -a/2
    ri = x > a/2
 
    psi0 = Psi0_gwp(x,m=m,sig=sig_x,k=k,x0=x0)
    R0   = dx*np.sum(np.abs(psi0[li])**2)
    T0   = dx*np.sum(np.abs(psi0[ri])**2)
    for i in range(max_iter):
        # if i % 10000 == 0: print(i)
        psi = U_psi_ft(dt,x,psi0,m=m,V=lambda t: V(t,a=a,V0=V0))
        rho = np.abs(psi)**2
        
        R = dx*np.sum(rho[li])
        T = dx*np.sum(rho[ri])
        
        adiff = max(abs(R-R0),abs(T-T0))
        rdiff = max(abs(1-R/(R0+1e-100)),abs(1-T/(T0+1e-100)))

        cond1 = (2*x0 + k*i*dt/m > 0)
        cond2 = (rdiff < rtol or adiff < atol)
        if cond1 and cond2:
            # print(i,rdiff<rtol,adiff<atol)
            break
        else:
            psi0  = psi.copy()
            R0,T0 = R.copy(),T.copy()
    
    # t = i*dt
    # print(t**2/(4*sig_x**4*m**2))
            
    return np.array([R,T])


m    = 1
V0,a = 0.1,10
k    = np.linspace(0.1,1,50)


start = time.time()
results = Parallel(n_jobs=4)(delayed(get_RT)(_,m=m,M2=500,dt=0.1,V0=V0,a=a) for _ in k)
R,T = np.array(results).T
end = time.time()
print(end - start)


nrows,ncols = 2,1
fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7,7),gridspec_kw={'height_ratios':[6,2]})

ke  = np.linspace(0,k[-1],1000)
Te_ = np.array([B2(_,m=m,V0=V0,a=a) for _ in ke])
ax[0].plot(ke,Te_,'r-',label=r'${\rm exact}$')
ax[0].plot(k,T,color='k',ls='None',marker='.',markersize=10,label=r'$\rm numeric$')

# ax[0].semilogy()
ax[0].set_ylabel(r'$|B(k)|^2$',size=30)
ax[0].tick_params(axis='both',which='major',labelsize=20,direction='in')
ax[0].tick_params(axis='x',which='major',labelsize=0)
ax[0].legend(fontsize=20,frameon=False,loc='lower right')
text =(
    r'\begin{align*}'
    rf'V_0 &= {V0} \\'
    rf'a &= {a} \\'
    rf'm &= {m}'
    r'\end{align*}'
)
ax[0].text(s=text,size=20,x=0.05,y=0.95,ha='left',va='top',transform=ax[0].transAxes)

#####

Te_ = np.array([B2(_,m=m,V0=V0,a=a) for _ in k])
ax[1].plot(k,T/(Te_+1e-100),color='k',ls='-',marker='.',markersize=10)
ax[1].axhline(1,color='k',ls='-.',alpha=0.5)

ax[1].tick_params(axis='both',which='major',labelsize=20,direction='in')
ax[1].set_xlabel(r'$k/m$',size=30)
ax[1].set_ylabel(r'$\rm ratio$',size=30)
ax[1].set_xlim(-0.05,1.05)
ax[1].set_ylim(0.8,1.2)

fig.align_labels()

plt.tight_layout()
plt.show()
fig.savefig(r'square_barrier.pdf',bbox_inches='tight')