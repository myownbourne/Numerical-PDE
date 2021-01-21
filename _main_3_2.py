from Ritz_Galerkin import *
import matplotlib.pyplot as plt

eps=1e-4
print('--Please wait a moment:)--')
def f(x):
    return -x

def objfun(x):
    return np.sin(x)/np.sin(1)-x

def phi(x,n):
    center=n*h
    if x>=center-h and x<center:
        return (x-(center-h))/h
    elif x>=center and x<center+h:
        return (center+h-x)/h
    else:
        return 0

def dphi(x,n):
    center=n*h
    if x>=center-h and x<center:
        return 1/h
    elif x>=center and x<center+h:
        return -1/h
    else:
        return 0
N=5
x_range=[0,1]
h=(x_range[1]-x_range[0])/(N+1)
c=Ritz_Galerkin(f,[0,1],N,phi,dphi)
test_x=np.arange(0,1+eps,0.01)
test_y1=np.zeros_like(test_x)
for i in range(len(test_x)):
    test_y1[i]=u_result(test_x[i],c,phi)

N=10
x_range=[0,1]
h=(x_range[1]-x_range[0])/(N+1)
c=Ritz_Galerkin(f,[0,1],N,phi,dphi)
test_x=np.arange(0,1+eps,0.01)
test_y2=np.zeros_like(test_x)
for i in range(len(test_x)):
    test_y2[i]=u_result(test_x[i],c,phi)

N=15
x_range=[0,1]
h=(x_range[1]-x_range[0])/(N+1)
c=Ritz_Galerkin(f,[0,1],N,phi,dphi)
test_x=np.arange(0,1+eps,0.01)
test_y3=np.zeros_like(test_x)
for i in range(len(test_x)):
    test_y3[i]=u_result(test_x[i],c,phi)

test_y4=objfun(test_x)
Ns=[5,10,15]
max_errors=[]
max_errors.append(np.max(np.abs(test_y1-test_y4)))
max_errors.append(np.max(np.abs(test_y2-test_y4)))
max_errors.append(np.max(np.abs(test_y3-test_y4)))
max_errors=['{:.4e}'.format(i) for i in max_errors] 
print('n:        ',Ns)
print('max_error:',max_errors)

fig=plt.figure()
fig.suptitle('Visualization of FEM')
fig.set_size_inches(16,4)
ax1=fig.add_subplot(1,4,1) 
ax2=fig.add_subplot(1,4,2)
ax3=fig.add_subplot(1,4,3) 
ax4=fig.add_subplot(1,4,4)

ax1.plot(test_x,test_y1,'blue')
ax2.plot(test_x,test_y2,'blue')
ax3.plot(test_x,test_y3,'blue')
ax4.plot(test_x,test_y4,'red')
ax1.set_title('FEM(n=5)')
ax2.set_title('FEM(n=10)')
ax3.set_title('FEM(n=15)')
ax4.set_title('Objective function')
plt.show()

