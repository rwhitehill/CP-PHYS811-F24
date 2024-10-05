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
    
    sets = {
        1: [lambda t: 1/(1-0.998*t**2),[0,1]],
        2: [lambda t: t*np.sin(30*t)*np.cos(50*t),[0,2*np.pi]],
        3: [lambda t: t/(np.exp(t) - 1),[0,1]],
        4: [lambda t: t*np.sin(1/t),[0,1]]
    }
    
    N = 100
    results = {}
    for i in sets:
        f,ab = sets[i]
        results[i] = {}
        results[i]['trap']  = trap_quad(f,*ab,N=N)
        results[i]['rom']   = rom_quad(f,*ab,N=int(N/2))
        results[i]['gauss'] = gauss_quad(f,*ab)
        results[i]['scipy'] = quad(f,*ab)
        
    print(results)
        