import numpy as np
from Poission_Equation import *
import matplotlib.pyplot as plt

def inv_transform(x,y,M):
    return round((y-1)*M+x-1)

def Biharmonic_Equation(a,b,f,g1,g2,x_range,y_range,h1,h2):
    """
    Solving:
    tri^2 u-a*tri u+bu=f in the area
    u=g on the border
    tri u=g2 on the border
    """
    ##########
    """
    Solving the first Poission equation:
    mu*tri v-b*v=f in the area
    v=1/mu*g2-g1 on the border
    where mu=0.5*(a+sqrt(a**2-4*b))
    """
    mu=0.5*(a+np.sqrt(a**2-4*b))
    def g(x,y):
        return g2(x,y)/mu-g1(x,y)
    V,M1,N1,A,B=Poission_Equation(mu,b,f,g,x_range,y_range,h1,h2)
    """
    Solving the second Poission equation:
    1/mu*tri u-u=v in the area
    u=g1 on the border
    """
    def f2(x,y):
        x_index=(x-x_range[0])/h1
        y_index=(y-y_range[0])/h2
        return V[inv_transform(x_index,y_index,M1)]
    U,M2,N2,A2,B2=Poission_Equation(1/mu,1,f2,g1,x_range,y_range,h1,h2)
    return U,M2,N2

def error(a,b,f,g1,g2,x_range,y_range,h1,h2,obj):
    U,M2,N2=Biharmonic_Equation(a,b,f,g1,g2,x_range,y_range,h1,h2)
    max_diff=0
    total_diff=0
    for i in range(len(U)):
        x,y=transform(i,M2)
        pi=obj(x_range[0]+x*h1,y_range[0]+y*h2)
        diff=U[i]-pi
        total_diff+=abs(diff)
        if abs(diff)>max_diff:
            max_diff=abs(diff)
    return max_diff,total_diff/len(U)
