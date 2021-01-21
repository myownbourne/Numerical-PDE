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
    return 100*x

def ux02(x):
    return 100*(1-x)

x_range=[0,1]
t_range=[0,1]
a=1
eps=1e-8
h=1/50
tau=1/100

result1=Parabolic_Equation_CN(h,tau,a,f,x_range,t_range,u0t,u1t,ux0,2,1)
result2=Parabolic_Equation_CN(h,tau,a,f,x_range,t_range,u0t,u1t,ux02,1,2)

#visualization
xx1=np.arange(x_range[0],x_range[1]+eps,h)
yy1=np.arange(t_range[0],t_range[1]+eps,tau)
X1,Y1=np.meshgrid(xx1, yy1)
Z1=result1

xx2=np.arange(x_range[0],x_range[1]+eps,h)
yy2=np.arange(t_range[0],t_range[1]+eps,tau)
X2,Y2=np.meshgrid(xx2,yy2)
Z2=result2

fig=plt.figure()
fig.suptitle('visualization(h=1/50, tau=1/100)')
fig.set_size_inches(10,4)
ax1=fig.add_subplot(121,projection='3d')  
ax2=fig.add_subplot(122,projection='3d')
ax1.plot_surface(X1,Y1,Z1,cmap='rainbow')
ax2.plot_surface(X2,Y2,Z2,cmap='rainbow')
ax1.set_title('Problem 3')
ax2.set_title('Problem 4')
plt.show()
