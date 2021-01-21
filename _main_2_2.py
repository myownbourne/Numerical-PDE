from Parabolic_Equation_forward import *
from Parabolic_Equation_backward import *
from Parabolic_Equation_CN import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

k=1000

def f(x,t):
    return k*np.sin(np.pi*x)

def u0t(t):
    return 0

def u1t(t):
    return 0

def ux0(x):
    return 0

def objfun(x,t):
    return k*(1-np.exp(-1*np.pi**2*t))*np.sin(np.pi*x)/np.pi**2

x_range=[0,1]
t_range=[0,1]
a=1
eps=1e-4

#####
h=1/20
tau=1/400 
r=1
avg_errors=[]
max_errors=[]
x_span=x_range[1]-x_range[0]
t_span=t_range[1]-t_range[0]
N=int(x_span/h)
M=int(t_span/tau)
true=np.zeros((M+1,N+1))
for i in range(M+1):
    for j in range(N+1):
        true[i,j]=objfun(x_range[0]+j*h,t_range[0]+i*tau)
result=Parabolic_Equation_forward(h,tau,a,f,x_range,t_range,u0t,u1t,ux0,1,1)
for i in range(100):
    avg_error=np.mean(np.abs(true[i]-result[i]))
    avg_errors.append(avg_error)
    max_errors.append(np.max(np.abs(true[i]-result[i])))
level=[0,1,2,3,4,5,10,30,50]
result1=[]
result2=[]
for i in level:
    result1.append(avg_errors[i])
    result2.append(max_errors[i])
result11=['{:.4f}'.format(i) for i in result1]
result22=['{:.4f}'.format(i) for i in result2]  
print('tau:',tau,' ,h:',h,' r:',r)
print('level:     ',level)
print('avg_errors:',result11)
print('max_errors:',result22)
print('norm:',np.linalg.norm(result-true))
print()

#####
h=1/20
tau=1/800 
r=1/2
avg_errors=[]
max_errors=[]
x_span=x_range[1]-x_range[0]
t_span=t_range[1]-t_range[0]
N=int(x_span/h)
M=int(t_span/tau)
true=np.zeros((M+1,N+1))
for i in range(M+1):
    for j in range(N+1):
        true[i,j]=objfun(x_range[0]+j*h,t_range[0]+i*tau)
result=Parabolic_Equation_forward(h,tau,a,f,x_range,t_range,u0t,u1t,ux0,1,1)
for i in range(100):
    avg_error=np.mean(np.abs(true[i]-result[i]))
    avg_errors.append(avg_error)
    max_errors.append(np.max(np.abs(true[i]-result[i])))
level=[0,1,2,3,4,5,10,30,50]
result1=[]
result2=[]
for i in level:
    result1.append(avg_errors[i])
    result2.append(max_errors[i])
result11=['{:.4f}'.format(i) for i in result1]
result22=['{:.4f}'.format(i) for i in result2]  
print('tau:',tau,' ,h:',h,' r:',r)
print('level:     ',level)
print('avg_errors:',result11)
print('max_errors:',result22)
print('norm:',np.linalg.norm(result-true))
print()

#####
h=1/20
tau=1/3200 
r=1/8
avg_errors=[]
max_errors=[]
x_span=x_range[1]-x_range[0]
t_span=t_range[1]-t_range[0]
N=int(x_span/h)
M=int(t_span/tau)
true=np.zeros((M+1,N+1))
for i in range(M+1):
    for j in range(N+1):
        true[i,j]=objfun(x_range[0]+j*h,t_range[0]+i*tau)
result=Parabolic_Equation_forward(h,tau,a,f,x_range,t_range,u0t,u1t,ux0,1,1)
for i in range(100):
    avg_error=np.mean(np.abs(true[i]-result[i]))
    avg_errors.append(avg_error)
    max_errors.append(np.max(np.abs(true[i]-result[i])))
level=[0,1,2,3,4,5,10,30,50]
result1=[]
result2=[]
for i in level:
    result1.append(avg_errors[i])
    result2.append(max_errors[i])
result11=['{:.4f}'.format(i) for i in result1]
result22=['{:.4f}'.format(i) for i in result2]  
print('tau:',tau,' ,h:',h,' r:',r)
print('level:     ',level)
print('avg_errors:',result11)
print('max_errors:',result22)
print('norm:',np.linalg.norm(result-true))
print()

#####
h=1/20
tau=1/200 
r=2
avg_errors=[]
max_errors=[]
x_span=x_range[1]-x_range[0]
t_span=t_range[1]-t_range[0]
N=int(x_span/h)
M=int(t_span/tau)
true=np.zeros((M+1,N+1))
for i in range(M+1):
    for j in range(N+1):
        true[i,j]=objfun(x_range[0]+j*h,t_range[0]+i*tau)
result=Parabolic_Equation_forward(h,tau,a,f,x_range,t_range,u0t,u1t,ux0,1,1)
for i in range(100):
    avg_error=np.mean(np.abs(true[i]-result[i]))
    avg_errors.append(avg_error)
    max_errors.append(np.max(np.abs(true[i]-result[i])))
level=[0,1,2,3,4,5,10,30,50]
result1=[]
result2=[]
for i in level:
    result1.append(avg_errors[i])
    result2.append(max_errors[i])
result11=['{:.4f}'.format(i) for i in result1]
result22=['{:.4f}'.format(i) for i in result2]  
print('tau:',tau,' ,h:',h,' r:',r)
print('level:     ',level)
print('avg_errors:',result11)
print('max_errors:',result22)
print('norm',np.linalg.norm(result-true))