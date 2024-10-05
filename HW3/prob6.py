import numpy as np
from scipy.integrate import quad
from main import *

if __name__ == '__main__':
    
    sets = {
        1: [lambda t: 1/(1-0.998*t**2),[0,1]],
        2: [lambda t: t*np.sin(30*t)*np.cos(50*t),[0,2*np.pi]],
        3: [lambda t: t/(np.exp(t) - 1),[0,1]],
        4: [lambda t: t*np.sin(1/t),[0,1]]
    }
    
    results = {}
    for i in sets:
        f,ab = sets[i]
        print(f'{i}: {adaptive_gauss_quad(f,*ab,n=100)}')