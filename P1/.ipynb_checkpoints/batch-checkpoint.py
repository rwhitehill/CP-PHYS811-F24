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
L = N*2*(1/6)

results = {'method 1': {'configs': [], 'U': []}, 'method 2': {'configs': [], 'U': []}}
for i in trange(100):
    P = L*np.random.uniform(size=(N,2))
    configs,U = minimize1(P,h,max_iter=int(1e3),track=False)
    results['method 1']['configs'].append(configs)
    results['method 1']['U'].append(U)
    
    configs,U = minimize(P,h,max_iter=int(1e6),track=False)
    results['method 2']['configs'].append(configs)
    results['method 2']['U'].append(U)
    
for _1 in results:
    for _2 in results[_1]:
        results[_1][_2] = np.array(results[_1][_2])

        
nrows,ncols=1,2
fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*ncols,5*nrows))

colors = {'method 1': 'r', 'method 2': 'c'}
for _ in results:
    idxs = np.argsort(results[_]['U'])
    configs = results[_]['configs'][idxs]
    U = results[_]['U'][idxs]
    counts,edges = np.histogram(U,bins=30,range=(-10,0),density=True)
    ax[0].stairs(counts,edges,color=colors[_])
    
    x,y = configs[0].T
    ax[1].scatter(x,y,marker='o',s=75,lw=2,color=colors[_],alpha=0.4,edgecolor=colors[_],label=r'$\rm %s$'%_) 
    
ax[0].set_xlabel(r'$U_{\rm min}$',size=30)
ax[0].set_ylabel(r'$\rm normalized~yield$',size=30)
ax[0].text(s=r'\boldmath $N = %d$'%N,size=30,x=0.95,y=0.95,va='top',ha='right',transform=ax[0].transAxes)

ax[1].set_xlabel(r'$x$',size=30)
ax[1].set_ylabel(r'$y$',size=30)

scale = 1.05*max(np.amax(np.abs(results['method 1']['configs'])),np.amax(np.abs(results['method 2']['configs'])))
ax[1].set_xlim(-scale,scale)
ax[1].set_ylim(-scale,scale)
ax[1].legend(frameon=False,fontsize=20,loc='upper left')

for i in range(2):
    ax[i].tick_params(axis='both',which='major',labelsize=20,direction='in')

plt.show()
fig.savefig(r'batch_min%d.pdf'%N,bbox_inches='tight')