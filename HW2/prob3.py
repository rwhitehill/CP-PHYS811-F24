import numpy as np
from scipy.integrate import quad

import matplotlib.pyplot as plt
plt.rcParams.update({
    'text.latex.preamble': r'\usepackage{amsmath}',
    'text.usetex': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica']
})


def mean_quad(f,a,b,N=100,err=False):
    x  = np.random.uniform(a,b,size=N)
    f_ = f(x)
    ave_f = np.sum(f_)/N
    res   = (b-a)*ave_f
    if err:
        ave_f2 = np.sum(f_**2)/N
        error  = np.sqrt((ave_f2 - ave_f)/N)
        return res,error
    else:
        return res

def rejection_quad(f,x_bounds,y_bounds,N=100,verb=0):
    x  = np.random.uniform(*x_bounds,size=N)
    y  = np.random.uniform(*y_bounds,size=N)
    f_ = f(x)
    
    cond1 = np.logical_and(y > 0,y < f_)
    cond2 = np.logical_and(y < 0, y > f_)
    r     = (np.sum(cond1) - np.sum(cond2))/N
    A_box = (x_bounds[1] - x_bounds[0])*(y_bounds[1] - y_bounds[0])
    
    if verb:
        # print(cond1)
        # print(cond2)
        print(f'rejection rate: {(np.sum(cond1)+np.sum(cond2))/N}')

    return r*A_box

if __name__ == '__main__':
    N = np.array([10,100,1000,10000,100000])

    funcs = [
        lambda t: np.sin(t),
        lambda t: 1/(1-0.998*t**2),
        lambda t: t*np.sin(12*t)*np.cos(24*t),
        lambda t: np.sin(1/t/(2-t))**2
    ]
    bounds = [
        [0,np.pi],
        [0,1],
        [0,2*np.pi],
        [0,2]
    ]
    boxes = [
        [0,1],
        [0,1/(1-0.998)],
        [-2*np.pi,2*np.pi],
        [0,1]
    ]

    results = {}
    results['mean'] = []
    results['rejection'] = []
    results['exact'] = []

    for i in range(4):
        func = funcs[i]
        results['mean'].append([mean_quad(func,*bounds[i],N=_) for _ in N])
        results['rejection'].append([rejection_quad(func,bounds[i],boxes[i],N=_,verb=1) for _ in N])

        results['exact'].append(quad(func,*bounds[i])[0])

        print()
        
    nrows,ncols=1,4
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*ncols,5*nrows))

    expr = [
        r'$\displaystyle \int_{0}^{\pi} \sin{x} {\rm d}{x}$',
        r'$\displaystyle \int_{0}^{1} \frac{{\rm d}{x}}{1-0.998 x^2}$',
        r'$\displaystyle \int_{0}^{2 \pi} x \sin{12 x} \cos{24 x} {\rm d}{x}$',
        r'$\displaystyle \int_{0}^{2} \sin^2\Big[ \frac{1}{x(2-x)} \Big] {\rm d}{x}$'
    ]
    for i in range(4):
        temp = results['mean'][i]
        ax[i].plot(N,temp,marker='o',ls='--',alpha=0.8,color='r')

        temp = results['rejection'][i]
        ax[i].plot(N,temp,marker='o',ls='--',alpha=0.8,color='b')

        ax[i].axhline(y=results['exact'][i],color='k',ls='--',alpha=0.5)

        ax[i].tick_params(axis='both',which='major',direction='in',labelsize=20)
        ax[i].set_xlabel(r'$N$',size=30)
        ax[i].semilogx()

        ax[i].text(s=expr[i],x=0.95,y=0.95,va='top',ha='right',transform=ax[i].transAxes,size=30)

    plt.tight_layout()
    # plt.show()
    fig.savefig(r'prob3.pdf',bbox_inches='tight')

