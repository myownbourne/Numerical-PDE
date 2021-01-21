import numpy as np
from Conjugate_Gradient import *
import time

def transform(i,M):
    r=(i+1)%M
    if r==0:
        x=M
        y=(i+1-r)/M
    else: 
        x=r
        y=(i+1-r)/M+1
    return round(x),round(y)

def inv_transform(x,y,M):
    return (y-1)*M+x-1

def Poission_Equation(a,b,f,g,x_range,y_range,h1,h2):
    """
    Solving:
    a*tri u-b*u=f in the area
    u=g on the border
    """
    x_span=x_range[1]-x_range[0]
    y_span=y_range[1]-y_range[0]
    M=round(x_span/h1)-1
    N=round(y_span/h2)-1
    A=np.zeros((M*N,M*N))
    B=np.zeros((M*N))

    for i in range(M*N):
        x,y=transform(i,M)
        A[i][i]=-2*a/h1**2-2*a/h2**2-b
        B[i]=f(x_range[0]+h1*x,y_range[0]+h2*y)

        if x==1:
            B[i]-=a*g(x_range[0],y_range[0]+y*h2)/h1**2
        else:
            A[i][i-1]=a/h1**2

        if x==M:
            B[i]-=a*g(x_range[1],y_range[0]+y*h2)/h1**2
        else:
            A[i][i+1]=a/h1**2

        if y==1:
            B[i]-=a*g(x_range[0]+x*h1,y_range[0])/h2**2
        else:
            A[i][i-M]=a/h2**2

        if y==N:
            B[i]-=a*g(x_range[0]+x*h1,y_range[1])/h2**2
        else:
            A[i][i+M]=a/h2**2
    #U=np.linalg.solve(A,B)
    U=Conjugate_Gradient(A,B)
    return U,M,N,A,B








