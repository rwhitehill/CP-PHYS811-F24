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



def get_U_fde(dt,x,V=lambda t: np.full(t.shape[0],0),m=1):
    dx = x[1] - x[0]
    N  = x.shape[0]
    H  = -(np.diag(np.full(N-1,1),k=-1) - np.diag(np.full(N,2),k=0) + np.diag(np.full(N-1,1),k=1))/2/m/dx**2 + np.diag(V(x),k=0)
    H[0][:2]   = np.array([1,0])
    H[-1][-2:] = np.array([0,1])
    print(H)
    
    U = np.eye(N) - complex(0,dt)*H
    return U

def get_U_fdi(dt,x,V=lambda t: np.full(t.shape[0],0),m=1):
    dx = x[1] - x[0]
    N  = x.shape[0]
    H  = -(np.diag(np.full(N-1,1),k=-1) - np.diag(np.full(N,2),k=0) + np.diag(np.full(N-1,1),k=1))/2/m/dx**2 + np.diag(V(x),k=0)
    H[0][:2]   = np.array([1,0])
    H[-1][-2:] = np.array([0,1])
    
    U = np.eye(N) + complex(0,dt)*H
    return U


def Psi0_gwp(x,m=1,sig=1,k=0,x0=0):
    return (2*np.pi*sig**2)**(-1/4)*np.exp(-(x-x0)**2/4/sig**2 + complex(0,1)*k*x)
    
def Psi2_gwp(x,t,m=1,sig=1,k=0,x0=0):
    return 1/np.sqrt(2*np.pi*sig**2)*2*sig**2*m/np.sqrt(4*sig**4*m**2+t**2)*np.exp(-2*m**2*sig**2*(x-x0-k*t/m)**2/(4*sig**4*m**2+t**2))


nrows,ncols = 1,1
fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*ncols,5*nrows))

rho0 = np.abs(psi0)**2
ax.plot(x,rho0,'k-',label=r'$t=0~({\rm exact})$')

temp = Psi2_gwp(x,Nt*dt,x0=x0)
ax.plot(x,temp,'k-.',label=r'$t=%.1g~({\rm exact})$'%(Nt*dt))

psi = (Ue**Nt)@psi0
rho = np.abs(psi)**2
ax.plot(x,rho,'r.',label=r'$\rm explicit$')

psi = np.linalg.solve((Ui**Nt),psi0)
rho = np.abs(psi)**2
ax.plot(x,rho,'c.',label=r'$\rm implicit$')

ax.tick_params(axis='both',which='major',direction='in',labelsize=20)
ax.set_xlabel(r'$x$',size=30)
ax.set_ylabel(r'$|\Psi(x,t)|^2$',size=30)
ax.legend(fontsize=20,frameon=False,loc='upper right')
ax.set_xlim()

plt.show()
fig.savefig(r'finite_differences-evolution.pdf',bbox_inches='tight')
