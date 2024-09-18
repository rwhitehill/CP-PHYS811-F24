import numpy as np
from scipy.optimize import root_scalar,fsolve
from scipy.integrate import quad

import matplotlib.pyplot as plt
plt.rcParams.update({
    'text.latex.preamble': r'\usepackage{amsmath}',
    'text.usetex': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica']
})


def get_mom(sample,n=1):
    return np.sum(sample**n)/sample.shape[0]

def get_nn_corr(sample,k=1):
    return np.sum(sample[:-k]*sample[k:])/sample.shape[0]



if __name__ == '__main__':

    rng = np.random.default_rng(seed=12345)

    N = 100000
    sample = np.array([rng.random() for i in range(N)])


    # one sequence uniformity
    nrows,ncols = 1,1
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*ncols,5*nrows))

    bins = 50
    counts,edges = np.histogram(sample,bins=bins,density=False)
    ax.stairs(counts,edges)

    ax.set_xlabel(r'$\rm uniform~RV$',size=30)
    ax.set_ylabel(r'$\rm normalized~yield$',size=30)
    ax.tick_params(which='major',axis='both',labelsize=20,direction='in')
    ax.axhline(N/bins,color='k',ls='--')

    plt.tight_layout()
    # plt.show()
    fig.savefig(r'prob1_uniformity.pdf',bbox_inches='tight')

    
    # parking lot
    N = 1000
    sample1 = np.array([rng.random() for i in range(N)])
    sample2 = np.array([rng.random() for i in range(N)])

    nrows,ncols = 1,1
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*ncols,5*nrows))

    ax.scatter(sample1,sample2,alpha=0.5)
    ax.set_xlabel(r'\rm RV~1',size=30)
    ax.set_ylabel(r'\rm RV~2',size=30)
    ax.tick_params(axis='both',which='major',direction='in',labelsize=20)

    # plt.show()
    fig.savefig(r'prob1_parking-lot.pdf',bbox_inches='tight')

    
    # uniform sample 5th moment and near-neighbor correlation
    N = np.arange(10,10000,100)

    n = 5
    k = 5
    mom  = []
    corr = []
    for _ in N:
        temp = np.array([rng.random() for i in range(_)])
        mom.append(get_mom(temp,n=n))
        corr.append(get_nn_corr(temp,k=k))

    nrows,ncols=1,2
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*ncols,5*nrows))

    ax[0].plot(N,mom)
    ax[0].axhline(1/(k+1),color='k',ls='--')

    ax[1].plot(N,corr)
    ax[1].axhline(1/4,color='k',ls='--')

    for i in range(2):
        ax[i].set_xlabel(r'$\rm sample~size$',size=30)
        ax[i].tick_params(axis='both',which='major',direction='in',labelsize=20)
        ax[0].set_ylabel(r'$\langle x^{%d} \rangle$'%n,size=30)
        ax[1].set_ylabel(r'$C(%d)$'%k,size=30)

    plt.tight_layout()
    # plt.show()
    fig.savefig(r'prob1_mom_corr.pdf',bbox_inches='tight')




