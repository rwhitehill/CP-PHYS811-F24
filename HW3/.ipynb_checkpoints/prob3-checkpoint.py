import numpy as np
from scipy.integrate import quad
from main import *

import matplotlib.pyplot as plt
plt.rcParams.update({
    'text.latex.preamble': r'\usepackage{amsmath}',
    'text.usetex': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica']
})

if __name__ == '__main__':
    
    func = lambda t: np.exp(-t**2)/(1+t**2)
    print(gauss_quad_improper(func,0,np.inf,n=10))
    print(quad(func,0,np.inf))
    print()

    func = lambda t: t*np.sin(t)/(1 + t**2)
    print(gauss_quad_improper(func,0,np.inf,n=10))
    print(quad(func,0,np.inf))
    print()

    func = lambda t: np.exp(-np.sqrt(t))*np.cos(2*t)
    print(gauss_quad_improper(func,0,np.inf,n=1000))
    print(quad(func,0,np.inf))
    print()