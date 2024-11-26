import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({
    'text.latex.preamble': r'\usepackage{amsmath}',
    'text.usetex': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica']
})
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec



Nx = 100
x = np.linspace(0,1,Nx)
dx = x[1] - x[0]
a = 1
b = 0.25

dt = b*dx**2/a
Nt = 10000
t = dt*np.arange(Nt)

X,T = np.meshgrid(x,t)



f0 = np.zeros(Nx)
f0[1:-1] = 100

f = [f0]
for i in range(Nt-1):
    temp = f[-1].copy()
    temp[1:-1] += b*(temp[:-2] - 2*temp[1:-1] + temp[2:])
    f.append(temp)
    
    
for _ in f[::1000]:
    plt.plot(x,_)
    
plt.show()



nrows,ncols = 1,1
fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*nrows,5*ncols))

img = ax.contourf(X,T,f,cmap='coolwarm')#,vmin=0,vmax=1)
cbar = fig.colorbar(img)
cbar.ax.tick_params(labelsize=20)
# ticks = [0,0.2,0.4,0.6,0.8,1]
# cbar.ax.set_yticks(ticks,[r'$%.1g$'%_ for _ in ticks])

ax.tick_params(axis='both',which='major',direction='out',labelsize=20)
ax.set_title(r'$\displaystyle T(x,t)$',size=30,pad=10)
ax.set_xlabel(r'$x/h$',size=30)
ax.set_ylabel(r'$t$',size=30)

plt.show()
fig.savefig(r'part3-sol_stable.pdf',bbox_inches='tight')



def f_exact(x,t,f0=100,a=1,N=100):
    n   = 2*np.arange(N) + 1
    A_n = 4*f0/(n*np.pi)
    f_n = A_n * np.sin(n*np.pi*x) * np.exp(-a*n**2*np.pi**2*t)
    return np.sum(f_n)

f_exact_ = np.vectorize(f_exact)(X,T,f0=100,N=100)



nrows,ncols = 1,2
fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*ncols,5*nrows))

img0 = ax[0].contourf(X,T,f,cmap='coolwarm')

ax[0].tick_params(axis='both',which='major',direction='out',labelsize=20)
# ax[0].set_title(r'$\displaystyle V/V_0$',size=30,pad=10)
ax[0].set_xlabel(r'$x/h$',size=30)
ax[0].set_ylabel(r'$y/h$',size=30)

img1 = ax[1].contourf(X,T,f-f_exact_,cmap='Reds')

ax[1].tick_params(axis='both',which='major',direction='out',labelsize=20)
# ax[1].set_title(r'$\displaystyle V/V_0$',size=30,pad=10)
# ax[1].set_xlabel(r'$x/h$',size=30)
ax[1].set_ylabel(r'$y/h$',size=30)

plt.tight_layout()

ticks = [0,20,40,60,80,100]
cbaxes = inset_axes(ax[0], width="70%", height="5%", loc='lower center') 
cbar = fig.colorbar(img0,cax=cbaxes,orientation='horizontal',ticks=ticks)
cbaxes.xaxis.set_ticks_position('top')
cbar.ax.tick_params(labelsize=20)
cbar.ax.set_xticklabels([r'$%.3g$'%_ for _ in ticks])
cbar.ax.set_title(r'$T(x,y)$',size=30)

ticks = [-10,-5,0,5,10]
# print(np.amax)
print(np.amax(f - f_exact_))
cbaxes = inset_axes(ax[1], width="70%", height="5%", loc='lower center') 
cbar = fig.colorbar(img1,cax=cbaxes,orientation='horizontal',ticks=ticks)
cbaxes.xaxis.set_ticks_position('top')
cbar.ax.tick_params(labelsize=20)
# cbar.ax.set_xticklabels([r'$%.1g$'%_ for _ in ticks])
cbar.ax.set_title(r'$(T_{\rm numeric} - T_{\rm analytic})$',size=30)

plt.show()
fig.savefig(r'part3-sol.pdf')