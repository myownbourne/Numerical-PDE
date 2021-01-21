from Ritz_Galerkin import *
import matplotlib.pyplot as plt

def f(x):
    return -x

def objfun(x):
    return np.sin(x)/np.sin(1)-x

def phi(x,n):
    return x**n*(1-x)

def dphi(x,n):
    return n*x**(n-1)-(n+1)*x**n

eps=1e-4
c=Ritz_Galerkin(f,[0,1],4,phi,dphi)
test_x=np.arange(0,1+eps,0.01)
test_y1=np.zeros_like(test_x)
for i in range(len(test_x)):
    test_y1[i]=u_result(test_x[i],c,phi)
test_y2=objfun(test_x)

fig=plt.figure()
ax1=fig.add_subplot(1,2,1) 
ax2=fig.add_subplot(1,2,2)

ax1.plot(test_x,test_y1,'blue')
ax2.plot(test_x,test_y2,'red')
ax1.set_title('Ritz_Galerkin(n=4)')
ax2.set_title('Objective function')


ns=[2,3,4,5]
max_error=[]
for n in ns: 
    c=Ritz_Galerkin(f,[0,1],n,phi,dphi)
    test_x=np.arange(0,1,0.01)
    test_y1=np.zeros_like(test_x)
    for i in range(len(test_x)):
        test_y1[i]=u_result(test_x[i],c,phi)
    test_y2=objfun(test_x)
    max_error.append(np.max(np.abs(test_y1-test_y2)))

max_error2=['{:.4e}'.format(i) for i in max_error]  
print('n:        ',ns)
print('max_error:',max_error2)
plt.show()
