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
eps=1e-8

#Convergence order of forward difference O(tau+h^2) a*tau/h^2<=0.5
#tau
h=1/10
taus=[1/200,1/210,1/220,1/230,1/240,1/250]
max_error=[]
for tau in taus:
    x_span=x_range[1]-x_range[0]
    t_span=t_range[1]-t_range[0]
    N=int(x_span/h)
    M=int(t_span/tau)
    true=np.zeros((M+1,N+1))
    for i in range(M+1):
        for j in range(N+1):
            true[i,j]=objfun(x_range[0]+j*h,t_range[0]+i*tau)
    result=Parabolic_Equation_forward(h,tau,a,f,x_range,t_range,u0t,u1t,ux0,1,1)
    error=np.abs(true-result)
    max_error.append(error.max())
taus2=['{:.4f}'.format(i) for i in taus]
max_error2=['{:.4f}'.format(i) for i in max_error]  
print('tau:  ',taus2)
print('error:',max_error2)
max_error=np.array(max_error)
max_error=np.log(max_error)
taus=np.array(taus)
taus=np.log(taus)

plt.figure(1)
plt.title('Convergence order of tau(forward difference)')
plt.plot(taus,max_error,'red')
plt.xlabel('log tau')
plt.ylabel('log error')
plt.scatter(taus,max_error)
print('Convergence order of tau(forward difference):','%.4f' % ((max_error[-1]-max_error[0])/(taus[-1]-taus[0])))
print()
#plt.show()


#h
tau=1/3000
hs=[1/3,1/5,1/7,1/9,1/20]
max_error=[]
for h in hs:
    x_span=x_range[1]-x_range[0]
    t_span=t_range[1]-t_range[0]
    N=int(x_span/h)
    M=int(t_span/tau)
    true=np.zeros((M+1,N+1))
    for i in range(M+1):
        for j in range(N+1):
            true[i,j]=objfun(x_range[0]+j*h,t_range[0]+i*tau)
    result=Parabolic_Equation_forward(h,tau,a,f,x_range,t_range,u0t,u1t,ux0,1,1)
    error=np.abs(true-result)
    max_error.append(error.max())
hs2=['{:.4f}'.format(i) for i in hs]
max_error2=['{:.4f}'.format(i) for i in max_error]  
print('h:    ',hs2)
print('error:',max_error2)
max_error=np.array(max_error)
max_error=np.log(max_error)
hs=np.array(hs)
hs=np.log(hs)

plt.figure(2)
plt.title('Convergence order of h(forward difference)')
plt.plot(hs,max_error,'red')
plt.xlabel('log h')
plt.ylabel('log error')
plt.scatter(hs,max_error)
print('Convergence order of h(forward difference):','%.4f' %((max_error[-1]-max_error[0])/(hs[-1]-hs[0])))
print()
#plt.show()

#Convergence order of backward difference O(tau+h^2)
#tau
h=1/1000
taus=[1/8,1/10,1/20,1/30]
max_error=[]
for tau in taus:
    x_span=x_range[1]-x_range[0]
    t_span=t_range[1]-t_range[0]
    N=int(x_span/h)
    M=int(t_span/tau)
    true=np.zeros((M+1,N+1))
    for i in range(M+1):
        for j in range(N+1):
            true[i,j]=objfun(x_range[0]+j*h,t_range[0]+i*tau)
    result=Parabolic_Equation_backward(h,tau,a,f,x_range,t_range,u0t,u1t,ux0,1,1)
    error=np.abs(true-result)
    max_error.append(error.max())
taus2=['{:.4f}'.format(i) for i in taus]
max_error2=['{:.4f}'.format(i) for i in max_error]  
print('tau:  ',taus2)
print('error:',max_error2)
max_error=np.array(max_error)
max_error=np.log(max_error)
taus=np.array(taus)
taus=np.log(taus)


plt.figure(3)
plt.title('Convergence order of tau(backward difference)')
plt.plot(taus,max_error,'red')
plt.xlabel('log tau')
plt.ylabel('log error')
plt.scatter(taus,max_error)
print('Convergence order of tau(backward difference):','%.4f' %((max_error[-1]-max_error[0])/(taus[-1]-taus[0])))
print()
#plt.show()


#h
tau=1/3000
hs=[1/3,1/5,1/7,1/9,1/20]
max_error=[]
for h in hs:
    x_span=x_range[1]-x_range[0]
    t_span=t_range[1]-t_range[0]
    N=int(x_span/h)
    M=int(t_span/tau)
    true=np.zeros((M+1,N+1))
    for i in range(M+1):
        for j in range(N+1):
            true[i,j]=objfun(x_range[0]+j*h,t_range[0]+i*tau)
    result=Parabolic_Equation_backward(h,tau,a,f,x_range,t_range,u0t,u1t,ux0,1,1)
    error=np.abs(true-result)
    max_error.append(error.max())
hs2=['{:.4f}'.format(i) for i in hs]
max_error2=['{:.4f}'.format(i) for i in max_error]  
print('h:    ',hs2)
print('error:',max_error2)
max_error=np.array(max_error)
max_error=np.log(max_error)
hs=np.array(hs)
hs=np.log(hs)

plt.figure(4)
plt.title('Convergence order of h(backward difference)')
plt.plot(hs,max_error,'red')
plt.xlabel('log h')
plt.ylabel('log error')
plt.scatter(hs,max_error)
print('Convergence order of h(backward difference):','%.4f' %((max_error[-1]-max_error[0])/(hs[-1]-hs[0])))
print()
#plt.show()

#Convergence order of CN O(tau^2+h^2) 
#tau
h=1/1000
taus=[1/8,1/10,1/20,1/30]
max_error=[]
for tau in taus:
    x_span=x_range[1]-x_range[0]
    t_span=t_range[1]-t_range[0]
    N=int(x_span/h)
    M=int(t_span/tau)
    true=np.zeros((M+1,N+1))
    for i in range(M+1):
        for j in range(N+1):
            true[i,j]=objfun(x_range[0]+j*h,t_range[0]+i*tau)
    result=Parabolic_Equation_CN(h,tau,a,f,x_range,t_range,u0t,u1t,ux0,1,1)
    error=np.abs(true-result)
    max_error.append(error.max())
taus2=['{:.4f}'.format(i) for i in taus]
max_error2=['{:.4f}'.format(i) for i in max_error]  
print('tau:  ',taus2)
print('error:',max_error2)
max_error=np.array(max_error)
max_error=np.log(max_error)
taus=np.array(taus)
taus=np.log(taus)


plt.figure(5)
plt.title('Convergence order of tau(CN)')
plt.plot(taus,max_error,'red')
plt.xlabel('log tau')
plt.ylabel('log error')
plt.scatter(taus,max_error)
print('Convergence order of tau(CN):','%.4f' %((max_error[-1]-max_error[0])/(taus[-1]-taus[0])))
print()
#plt.show()


#h
tau=1/4000
hs=[1/3,1/4,1/5,1/6]
max_error=[]
for h in hs:
    x_span=x_range[1]-x_range[0]
    t_span=t_range[1]-t_range[0]
    N=int(x_span/h)
    M=int(t_span/tau)
    true=np.zeros((M+1,N+1))
    for i in range(M+1):
        for j in range(N+1):
            true[i,j]=objfun(x_range[0]+j*h,t_range[0]+i*tau)
    result=Parabolic_Equation_CN(h,tau,a,f,x_range,t_range,u0t,u1t,ux0,1,1)
    error=np.abs(true-result)
    max_error.append(error.max())
hs2=['{:.4f}'.format(i) for i in hs]
max_error2=['{:.4f}'.format(i) for i in max_error]  
print('h:    ',hs2)
print('error:',max_error2)
max_error=np.array(max_error)
max_error=np.log(max_error)
hs=np.array(hs)
hs=np.log(hs)

plt.figure(6)
plt.title('Convergence order of h(CN)')
plt.plot(hs,max_error,'red')
plt.xlabel('log h')
plt.ylabel('log error')
plt.scatter(hs,max_error)
print('Convergence order of h(CN):','%.4f' %((max_error[-1]-max_error[0])/(hs[-1]-hs[0])))
print()

h=1/20
tau=1/1000
result1=Parabolic_Equation_forward(h,tau,a,f,x_range,t_range,u0t,u1t,ux0,1,1)
result2=Parabolic_Equation_backward(h,tau,a,f,x_range,t_range,u0t,u1t,ux0,1,1)
result3=Parabolic_Equation_CN(h,tau,a,f,x_range,t_range,u0t,u1t,ux0,1,1)

#visualization
xx1=np.arange(x_range[0],x_range[1]+eps,h)
yy1=np.arange(t_range[0],t_range[1]+eps,tau)
X1,Y1=np.meshgrid(xx1, yy1)
Z1=result1

xx2=np.arange(x_range[0],x_range[1]+eps,h)
yy2=np.arange(t_range[0],t_range[1]+eps,tau)
X2,Y2=np.meshgrid(xx2, yy2)
Z2=result2

xx3=np.arange(x_range[0],x_range[1]+eps,h)
yy3=np.arange(t_range[0],t_range[1]+eps,tau)
X3,Y3=np.meshgrid(xx2, yy2)
Z3=result3

xx4=np.arange(x_range[0],x_range[1],0.01)
yy4=np.arange(t_range[0],t_range[1],0.01)
X4,Y4=np.meshgrid(xx4,yy4)
Z_true=k*(1-np.exp(-1*np.pi*np.pi*Y4))*np.sin(np.pi*X4)/np.pi**2

fig=plt.figure()
fig.suptitle('visualization(h=1/20, tau=1/1000)')
fig.set_size_inches(14,4)
ax1=fig.add_subplot(141,projection='3d') 
ax2=fig.add_subplot(142,projection='3d')  
ax3=fig.add_subplot(143,projection='3d')
ax4=fig.add_subplot(144,projection='3d')
ax1.plot_surface(X1,Y1,Z1,cmap='rainbow')
ax2.plot_surface(X2,Y2,Z2,cmap='rainbow')
ax3.plot_surface(X3,Y3,Z3,cmap='rainbow')
ax4.plot_surface(X4,Y4,Z_true,cmap='rainbow')
ax1.set_title('forward difference')
ax2.set_title('backward difference')
ax3.set_title('CN')
ax4.set_title('objfun')
plt.show()
