import numpy as np
from tqdm import tqdm,trange

import matplotlib.pyplot as plt
plt.rcParams.update({
    'text.latex.preamble': r'\usepackage{amsmath}',
    'text.usetex': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica']
})

from main import *

h = 0.2
N = 4
L = N*2**(1/6)
P = L*np.random.uniform(size=(N,2))

results = {'method 1': {}, 'method 2': {}}
configs,U = minimize1(P,h,track=True)
results['method 1']['configs'] = configs
results['method 1']['U'] = U

configs,U = minimize(P,h,track=True)
results['method 2']['configs'] = configs
results['method 2']['U'] = U


nrows,ncols=1,2
fig,ax=plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*ncols,5*nrows))

x,y = configs[0].T
ax[0].scatter(x,y,marker='o',s=75,lw=2,color='k',alpha=0.4,edgecolor='k',label=r'$\rm initial$')

colors = {'method 1': 'r', 'method 2': 'c'}
Umin = min(results['method 1']['U'][-1],results['method 2']['U'][-1])
for _ in results:
    x,y = results[_]['configs'][-1].T
    ax[0].scatter(x,y,marker='o',s=75,lw=2,color=colors[_],alpha=0.4,edgecolor=colors[_],label=r'$\rm %s$'%_)
    
    U = results[_]['U']
    ax[1].plot(U-1.1*Umin,color=colors[_],label=r'$U_{\rm min} = %.4f$'%U[-1])

ax[0].legend(fontsize=20,loc='upper left',frameon=False)
ax[0].set_xlabel(r'$x$',size=30)
ax[0].set_ylabel(r'$y$',size=30)

scale = 1.05*max(np.amax(np.abs(results['method 1']['configs'])),np.amax(np.abs(results['method 2']['configs'])))
ax[0].set_xlim(-scale,scale)
ax[0].set_ylim(-scale,scale)


ax[1].set_xlabel(r'$\rm iteration$',size=30)
ax[1].set_ylabel(r'$U - 1.1 U_{\rm min}$',size=30)
ax[1].semilogy()
ax[1].legend(frameon=False,fontsize=20)

for i in range(2):
    ax[i].tick_params(axis='both',which='major',direction='in',labelsize=20)
ax[1].tick_params(axis='both',which='major',direction='in',labelsize=20)

plt.tight_layout()
plt.show()
fig.savefig(r'track_min%d.pdf'%N,bbox_inches='tight')
