import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def composite_integration_rule(a,b,f,h):
    T=0
    T+=f(a)
    T+=f(b)
    n=int((b-a)/h)
    for i in range(1,n+1):
        T+=2*f(a+(i*h))
    T=T*h/2
    return T

def Simpson_rule(a,b,f,h):
    S=0
    n=int((b-a)/h)
    for i in range(n):
        xk=a+i*h
        xkp1=a+(i+1)*h
        S+=(h/6)*(f(xk)+4*f((xk+xkp1)/2)+f(xkp1))
    return S

def Romberg(a,b,f,h):
    T=0
    n=int((b-a)/h)
    T+=f(a)
    T+=f(b)
    for i in range(1,n):
        T+=2*f(a+i*h)
    for i in range(n):
        T+=2*f(a+(i+0.5)*h)
    T=T*h/4
    return T








