import numpy as np
from scipy.integrate import quad
from main import *

if __name__ == '__main__':
    
    print(gauss_quad(lambda x: gauss_quad(lambda y: np.sin(x**2 + y**2),0,2,n=100),-1,1,n=100))
    
    def f(x,y):
        return 1/np.sqrt(x+y)/(1+x+y)**2

    I1 = lambda x: quad(f,0,1-x,args=(x,))[0]
    print(quad(I1,0,1)[0])