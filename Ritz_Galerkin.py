from Numerical_Integration import *
import numpy as np

def Bilinear_functional(i,j,phi_i,phi_j,dphi_i,dphi_j,x_range):
    def g(x):
        return -1*dphi_i(x,i)*dphi_j(x,j)+phi_i(x,i)*phi_j(x,j)
    result=Romberg(x_range[0],x_range[1],g,0.001)
    return result

def Inner_product(i,phi_i,f,x_range):
    def g(x):
        return phi_i(x,i)*f(x)
    result=Romberg(x_range[0],x_range[1],g,0.001)
    return result

def Ritz_Galerkin(f,x_range,n,phi,dphi):
    """
    Solving:
    Lu=u''+u=f, x in x_range
    u(x_range[0])=u(x_range[1])=0

    n: Dimension of trial function space.
    phi: Basis function.
    """
    A=np.zeros((n,n))
    b=np.zeros(n)
    for i in range(n):
        for j in range(n):
            A[i][j]=Bilinear_functional(j+1,i+1,phi,phi,dphi,dphi,x_range)
    for k in range(n):
        b[k]=Inner_product(k+1,phi,f,x_range)
    c=np.linalg.solve(A,b)
    return c

def u_result(x,c,phi):
    n=len(c)
    result=0
    for i in range(n):
        result+=c[i]*phi(x,i+1)
    return result


