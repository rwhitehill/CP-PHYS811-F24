import numpy as np
from scipy.integrate import quad
from main import *

func = lambda t: t**(-1/3)
print(gauss_quad(func,0,1,n=100))

func = lambda t: 1 + 1/t
print(gauss_quad(func,-1,0,n=10) + gauss_quad(func,0,1,n=10))

func = lambda t: 1/((t-1)*(t**2+1))
print(gauss_quad_improper(func,-np.inf,1,n=100) + gauss_quad_improper(func,1,np.inf,n=100))
print(quad(func,-100,100)[0])