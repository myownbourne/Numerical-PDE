import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Poission_Equation import *
from Biharmonic_Equation import *
import math

a=2
b=1/2
x_range=[0,np.pi]
y_range=[0,np.pi]

def f(x,y):
    return (1+a+b)*(x*np.sin(y)+y*np.sin(x))

def g1(x,y):
    return x*np.sin(y)+y*np.sin(x)

def g2(x,y):
    return -1*np.sin(x)*y-x*np.sin(y)

def obj(x,y):
    return x*np.sin(y)+y*np.sin(x)

h2=np.pi/50
test_h=[np.pi/5,np.pi/7,np.pi/9,np.pi/11]
max_diff=[]
x_range=[0,math.pi]
y_range=[0,math.pi]
for h in test_h:
    md,ae=error(a,b,f,g1,g2,x_range,y_range,h,h2,obj)
    max_diff.append(md)
test_h2=['{:.4f}'.format(i) for i in test_h]
max_diff2=['{:.4f}'.format(i) for i in max_diff]  
print()
print('h1:   ',test_h2)
print('error:',max_diff2)

max_diff=np.array(max_diff)
test_h=np.array(test_h)
max_diff=np.log(max_diff)
test_h=np.log(test_h)

plt.figure(1)
plt.title('Degree of convergence of h1')
plt.plot(test_h,max_diff,'red')
plt.xlabel('log h1')
plt.ylabel('log error')
plt.scatter(test_h,max_diff)
print('Degree of convergence of h1:','%.4f' % ((max_diff[-1]-max_diff[0])/(test_h[-1]-test_h[0])))
print()

h1=np.pi/60
test_h=[np.pi/6,np.pi/8,np.pi/10,np.pi/12]
max_diff=[]
x_range=[0,math.pi]
y_range=[0,math.pi]
for h in test_h:
    md,ae=error(a,b,f,g1,g2,x_range,y_range,h1,h,obj)
    max_diff.append(md)
test_h2=['{:.4f}'.format(i) for i in test_h]
max_diff2=['{:.4f}'.format(i) for i in max_diff]  
print()
print('h2:   ',test_h2)
print('error:',max_diff2)

max_diff=np.array(max_diff)
test_h=np.array(test_h)
max_diff=np.log(max_diff)
test_h=np.log(test_h)

plt.figure(2)
plt.title('Degree of convergence of h2')
plt.plot(test_h,max_diff,'red')
plt.xlabel('log h2')
plt.ylabel('log error')
plt.scatter(test_h,max_diff)
print('Degree of convergence of h2:','%.4f' % ((max_diff[-1]-max_diff[0])/(test_h[-1]-test_h[0])))
#plt.show()
print()

h1=np.pi/30
h2=np.pi/30
U,M2,N2=Biharmonic_Equation(a,b,f,g1,g2,x_range,y_range,h1,h2)
eps=1e-5
xx1=np.arange(x_range[0]+h1,x_range[1]-eps,h1)
yy1=np.arange(y_range[0]+h2,y_range[1]-eps,h2)
X1,Y1=np.meshgrid(xx1, yy1)
m1=len(yy1)
n1=len(xx1)
Z1=np.zeros((m1,n1))
for i in range(m1):
    for j in range(n1):
        index=inv_transform(j+1,i+1,M2)
        Z1[i][j]=U[index]

xx2=np.arange(x_range[0],x_range[1],0.01)
yy2=np.arange(y_range[0],y_range[1],0.01)
X2,Y2=np.meshgrid(xx2,yy2)
Z_true=X2*np.sin(Y2)+Y2*np.sin(X2)

fig=plt.figure()
fig.suptitle('Visualization of problem 2(a=2, b=1/2, h1=pi/30, h2=pi/30)')
fig.set_size_inches(10,5)
ax1=fig.add_subplot(121,projection='3d') 
ax2=fig.add_subplot(122,projection='3d') 
ax1.plot_surface(X1,Y1,Z1,cmap='rainbow')
ax2.plot_surface(X2,Y2,Z_true,cmap='rainbow')
ax1.set_title('Numerical result')
ax2.set_title('Objective function')
plt.show()
