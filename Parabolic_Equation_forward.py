import numpy as np

def Parabolic_Equation_forward(h,tau,a,f,x_range,t_range,u0t,u1t,ux0,right_boundary_value_condition,left_boundary_value_condition):
    x_span=x_range[1]-x_range[0]
    t_span=t_range[1]-t_range[0]
    N=int(x_span/h)
    M=int(t_span/tau)
    result=np.zeros((M+1,N+1))
    for i in range(N+1):
        result[0,i]=ux0(x_range[0]+i*h)
    for i in range(M+1):
        result[i, 0]=u0t(t_range[0]+i*tau)
        result[i,-1]=u1t(t_range[0]+i*tau)
    r=tau/(h**2)
    C1=np.eye(N-1,k=1)
    C2=np.eye(N-1,k=-1)
    C=C1+C2
    A=(1-2*r*a)*np.eye(N-1)+r*a*C
    if right_boundary_value_condition==2:
        A[-1]=0
        A[-1][-1]=1-r*a
        A[-1][-2]=r*a
    if left_boundary_value_condition==2:
        A[0]=0
        A[0][0]=1-r*a
        A[0][1]=r*a
    for i in range(1,M+1):
        fh=np.zeros(N-1)
        for j in range(N-1):
            fh[j]=f(x_range[0]+(j+1)*h,t_range[0]+(i-1)*tau)
        result[i,1:-1]=A.dot(result[i-1,1:-1])+tau*fh
    if right_boundary_value_condition==2:
        result[:,-1]=result[:,-2]
    if left_boundary_value_condition==2:
        result[:,0]=result[:,1]
    return result

def Parabolic_Equation_forward2(h,tau,a,f,x_range,t_range,u0t,u1t,ux0,right_boundary_value_condition,left_boundary_value_condition):
    x_span=x_range[1]-x_range[0]
    t_span=t_range[1]-t_range[0]
    N=int(x_span/h)
    M=int(t_span/tau)
    result=np.zeros((M+1,N+1))
    for i in range(N+1):
        result[0,i]=ux0(x_range[0]+i*h)
    for i in range(M+1):
        result[i, 0]=u0t(t_range[0]+i*tau)
        result[i,-1]=u1t(t_range[0]+i*tau)
    r=tau/(h**2)
    C1=np.eye(N,k=1)
    C2=np.eye(N,k=-1)
    C=C1+C2
    A=(1-2*r*a)*np.eye(N)+r*a*C
    if right_boundary_value_condition==2:
        A[-1]=0
        A[-1][-1]=1-2*r*a
        A[-1][-2]=2*r*a
    for i in range(1,M+1):
        fh=np.zeros(N)
        for j in range(N):
            fh[j]=f(x_range[0]+(j+1)*h,t_range[0]+(i-1)*tau)
        result[i,1:]=A.dot(result[i-1,1:])+tau*fh
    return result