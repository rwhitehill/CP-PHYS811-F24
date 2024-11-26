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

def relax_f(f,X,Y,max_iter=int(1e5),rho=lambda t1,t2: np.full_like(t1,0),aeps=1e-20,reps=1e-6):
    dx = X[0,1] - X[0][0]
    dy = Y[1,0] - Y[0,0]
    
    rho_ = rho(X,Y)[1:-1,1:-1]
    for i in range(max_iter):
        f1 = f.copy()
        
        fm0 = f1[:-2,1:-1]
        fp0 = f1[2:,1:-1]
        f0m = f1[1:-1,:-2]
        f0p = f1[1:-1,2:]
        
        f1[1:-1,1:-1] = 0.5/(1/dx**2 + 1/dy**2)*((fm0 + fp0)/dx**2 + (f0m + f0p)/dy**2 + rho_)
        
        adiff = f1 - f
        rdiff = adiff/(f + 1e-100)
    
        f = f1.copy()
        
        if np.all(np.abs(adiff) < aeps) or np.all(np.abs(rdiff) < reps):
            print(i)
            break
            
    return f


def rho(x,y,x0=1/3,y0=1/3,sig=1,q=10):
    return q*np.exp(-((x-x0)**2 + (y-y0)**2)/2/sig**2)/(2*np.pi*sig**2)#360*(np.isclose(x,1/3))*(np.isclose(y,1/3))


xa,xb,Nx = 0,1,100
ya,yb,Ny = 0,1,100

x   = np.linspace(xa,xb,Nx)
y   = np.linspace(ya,yb,Ny)
X,Y = np.meshgrid(x,y)

f0 = np.zeros((Nx,Ny))
f0[0,:]  = np.sin(x)
f0[:,0]  = 0.5*np.sin(y)
f0[-1,:] = np.abs(np.sin(2*x))
f0[:,-1] = 1.2*np.sin(y)

dx = x[1] - x[0]
dy = y[1] - y[0]


nrows,ncols = 1,1
fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*nrows,5*ncols))

rho_ = rho(X,Y,x0=0.5,y0=0.5,sig=0.001,q=1)
img = ax.contourf(X,Y,rho_,cmap='viridis')#,vmin=0,vmax=1)
cbar = fig.colorbar(img)
cbar.ax.tick_params(labelsize=20)
# ticks = [0,0.2,0.4,0.6,0.8,1]
# cbar.ax.set_yticks(ticks,[r'$%.1g$'%_ for _ in ticks])

ax.tick_params(axis='both',which='major',direction='out',labelsize=20)
ax.set_title(r'$\displaystyle V/V_0$',size=30,pad=10)
ax.set_xlabel(r'$x/h$',size=30)
ax.set_ylabel(r'$y/h$',size=30)

plt.show()


xa,xb,Nx = 0,1,100
ya,yb,Ny = 0,1,100

x   = np.linspace(xa,xb,Nx)
y   = np.linspace(ya,yb,Ny)
X,Y = np.meshgrid(x,y)

f0 = np.zeros((Nx,Ny))
f0[0,:]  = np.sin(x)
f0[:,0]  = 0.5*np.sin(y)
f0[-1,:] = np.abs(np.sin(2*x))
f0[:,-1] = 1.2*np.sin(y)

dx = x[1] - x[0]
dy = y[1] - y[0]


f = relax_f(f0,X,Y,rho=lambda t1,t2: rho(t1,t2,x0=1/3,y0=1/3,sig=0.005,q=-1))


nrows,ncols = 1,1
fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*nrows,5*ncols))

img = ax.contourf(X,Y,f,cmap='RdBu_r')#,vmin=0,vmax=1)
cbar = fig.colorbar(img)
cbar.ax.tick_params(labelsize=20)
# ticks = [0,0.2,0.4,0.6,0.8,1]
# cbar.ax.set_yticks(ticks,[r'$%.1g$'%_ for _ in ticks])

ax.tick_params(axis='both',which='major',direction='out',labelsize=20)
ax.set_title(r'$\displaystyle V/V_0$',size=30,pad=10)
ax.set_xlabel(r'$x/h$',size=30)
ax.set_ylabel(r'$y/h$',size=30)

plt.show()
fig.savefig(r'part2_%d-N.pdf'%Nx,bbox_inches='tight')



nrows,ncols = 1,3
fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*ncols,6*nrows))

for i,N in enumerate([10,100,500]):
    
    xa,xb,Nx = 0,1,N
    ya,yb,Ny = 0,1,N

    x   = np.linspace(xa,xb,Nx)
    y   = np.linspace(ya,yb,Ny)
    X,Y = np.meshgrid(x,y)

    f0 = np.zeros((Nx,Ny))
    f0[0,:]  = np.sin(x)
    f0[:,0]  = 0.5*np.sin(y)
    f0[-1,:] = np.abs(np.sin(2*x))
    f0[:,-1] = 1.2*np.sin(y)
    f = relax_f(f0,X,Y,rho=lambda t1,t2: rho(t1,t2,x0=1/3,y0=1/3,sig=0.005,q=-1),max_iter=30000)
    
    img = ax[i].contourf(X,Y,f,cmap='RdBu_r')
    ax[i].tick_params(axis='both',which='major',direction='out',labelsize=20)
    ax[i].set_title(r'$N = %d$'%N,size=30,pad=10)
    ax[i].set_xlabel(r'$x/h$',size=30)
    
ax[0].set_ylabel(r'$y/h$',size=30)

plt.tight_layout()

ticks = [-0.6,-0.2,0.2,0.6,1]
cbaxes = inset_axes(ax[1], width="70%", height="5%", loc='lower center') 
cbar = fig.colorbar(img,cax=cbaxes,orientation='horizontal',ticks=ticks)
cbaxes.xaxis.set_ticks_position('top')
cbar.ax.tick_params(labelsize=20)
cbar.ax.set_xticklabels([r'$%.1g$'%_ for _ in ticks])
cbar.ax.set_title(r'$V(x,y)$',size=30)

plt.show()
fig.savefig(r'part2-N.pdf',bbox_inches='tight')
