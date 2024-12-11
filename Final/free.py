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


nrows,ncols = 1,1
fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*ncols,5*nrows))

m    = 10
sig  = 1
k    = 5
x0   = -5
args = (m,sig,k,x0)
x = np.linspace(-20,20,10000)
dx = x[1] - x[0]

psi0 = Psi0_gwp(x,*args)
# ax.plot(x,np.abs(psi0)**2,color='r',ls='-',lw=5,alpha=0.5,label=r'$t = 0$')


dt   = 0.01
t    = np.arange(0,20+dt,dt)
cmap = cm.inferno
norm = colors.Normalize(vmin=t[0],vmax=t[-1])

Psi = [psi0]
for _ in t[1:]:
    temp = U_psi_ft(dt,x,Psi[-1],m=m)
    Psi.append(temp)

for i in range(len(t)):
    _ = t[i]
    if np.isclose(_,np.round(_)) and np.round(_) % 5 == 0:
        ax.plot(x,Psi2_gwp(x,_,*args),color=cmap(norm(_)),ls='-',lw=5,alpha=0.3)#,label=r'$t = 0$')
        ax.plot(x,np.abs(Psi[i])**2,color=cmap(norm(_)),ls='-.')

# print(dx*np.sum(np.abs(Psi[0])**2))
# print(dx*np.sum(np.abs(Psi[-1])**2))

ax.set_xlabel(r'$x$',size=30)
ax.set_ylabel(r'$|\Psi(x,t)|^2$',size=30)
ax.tick_params(axis='both',which='major',labelsize=20,direction='in')
ax.set_xlim(-10,10)

ax.plot([],[],'k-',lw=5,label=r'$\rm exact$')
ax.plot([],[],'k-.',label=r'$\rm numeric$')
ax.legend(fontsize=20,loc='upper right',frameon=False)

cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax,orientation='vertical')
cbar.set_label(r'$t$',size=30,rotation=0,labelpad=15)
cbar.ax.tick_params(labelsize=20)
# cbar.set_ticks([0,1,2,3,4,5])

plt.show()
fig.savefig('gauss_wave-prop.pdf',bbox_inches='tight')