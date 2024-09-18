import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({
    'text.latex.preamble': r'\usepackage{amsmath}',
    'text.usetex': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica']
})

def transform_sample(N,F_inv):
    u = np.random.uniform(size=N)
    sample = F_inv(u)
    return sample

def rejection_sample(N,f,x_ext,y_ext,verb=0):
    x  = np.random.uniform(*x_ext,size=N)
    y  = np.random.uniform(*y_ext,size=N)
    f_ = f(x)
    diff = f_ - y
    sample = x[diff>=0]
    if verb:
        print(f'rejection rate: {1 - sample.shape[0]/N}')
    return sample

def metropolis_sample(N,f,x0=0,delta=1,bounds=[-np.inf,np.inf],verb=0):
    sample = [x0]
    f_     = [f(x0)]
    
    count = 0
    while len(sample) <= N:
        x1 = sample[-1] + delta*(2*np.random.uniform() - 1)
        
        if x1 < bounds[0] or x1 > bounds[1]:
            continue
            
        f1 = f(x1)
        r  = f1/f_[-1]
        u  = np.random.uniform()
        if u <= r:
            sample.append(x1)
            f_.append(f1)
        
        count += 1
    
    sample = np.array(sample)
    if verb:
        print(f'rejection rate: {(count - sample.shape[0])/N}')
    
    return sample

if __name__ == '__main__':
    
    sample_dict = {}
    N = 100000


    func_a = lambda t: np.exp(-t)*(t > 0)
    sample_dict['Poisson'] = {}
    sample_dict['Poisson']['transform'] = transform_sample(N,F_inv=lambda t: -np.log(1-t))
    sample_dict['Poisson']['rejection'] = rejection_sample(N,func_a,[0,10],[0,1],verb=1)
    sample_dict['Poisson']['metropolis'] = metropolis_sample(N,func_a,x0=1,delta=0.1,verb=1)

    print()

    func_b = lambda t: 1/(np.pi*(1+t**2))
    sample_dict['Cauchy'] = {}
    sample_dict['Cauchy']['transform'] = transform_sample(N,F_inv=lambda t: np.tan(np.pi*(t-0.5)))
    sample_dict['Cauchy']['rejection'] = rejection_sample(N,func_b,[-10,10],[0,1/np.pi],verb=1)
    sample_dict['Cauchy']['metropolis'] = metropolis_sample(N,func_b,x0=0,delta=1,verb=1)

    print()

    func_c = lambda t: np.exp(-t**2/2)/np.sqrt(2*np.pi)
    sample_dict['Gauss'] = {}
    # sample_dict['Gauss']['transform'] = transform_sample(N,F_inv=lambda t: np.tan(np.pi*(t-1)/2))
    sample_dict['Gauss']['rejection'] = rejection_sample(N,func_c,[-10,10],[0,1/np.sqrt(2*np.pi)],verb=1)
    sample_dict['Gauss']['metropolis'] = metropolis_sample(N,func_c,x0=0,delta=1,verb=1)
    
    
    nrows,ncols = 1,3
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*ncols,5*nrows))


    Range = {'Poisson': (0,5), 'Cauchy': (-5,5), 'Gauss': (-5,5)}
    colors = {'transform':'r','rejection':'b','metropolis':'g'}
    for i,dist in enumerate(sample_dict):
        for j,method in enumerate(sample_dict[dist]):
            sample = sample_dict[dist][method]
            counts,edges = np.histogram(sample,bins=50,range=Range[dist],density=True)
            ax[i].stairs(counts,edges,color=colors[method],lw=2,alpha=0.5,label=r'$\rm %s$'%method)

    x = np.linspace(1e-15,5,1000)
    ax[0].plot(x,func_a(x),'k--',alpha=0.5)

    x = np.linspace(-5,5,1000)
    ax[1].plot(x,func_b(x),'k--',alpha=0.5)
    ax[1].set_xlim(-5,5)

    x = np.linspace(-5,5,1000)
    ax[2].plot(x,func_c(x),'k--',alpha=0.5)
    ax[2].set_xlim(-5,5)


    ax[0].text(s=r'\boldmath $\rm Poisson$',x=0.95,y=0.95,size=30,va='top',ha='right',transform=ax[0].transAxes)
    ax[1].text(s=r'\boldmath $\rm Cauchy$',x=0.95,y=0.95,size=30,va='top',ha='right',transform=ax[1].transAxes)
    ax[2].text(s=r'\boldmath $\rm Gaussian$',x=0.95,y=0.95,size=30,va='top',ha='right',transform=ax[2].transAxes)

    ax[0].legend(fontsize=25,loc='center right',frameon=False)
    ax[0].set_ylabel(r'$f(x)$',size=30)
    for i in range(3):
        ax[i].tick_params(axis='both',which='major',direction='in',labelsize=20)
        ax[i].set_xlabel(r'$x$',size=30)


    plt.tight_layout()
    # plt.show()
    fig.savefig(r'prob2.pdf',bbox_inches='tight')
