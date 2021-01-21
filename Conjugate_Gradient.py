import numpy as np
np.random.seed(2021)

def fx(x):
	return 0.5*np.dot(x.T,np.dot(A,x))-np.dot(b.T,x)

def Conjugate_Gradient(A,b):
	print('---using conjugate gradient method to solve system of linear equations---')
	matrixSize=A.shape[0]
	x_0=np.ones(matrixSize)
	r_0=np.dot(A,x_0)-b
	p_0=(-1)*r_0

	r_k=r_0
	p_k=p_0
	x_k=x_0
	#print(x_k)

	for i in range(1000):
		alpha_k=(np.dot(r_k.T,r_k))/(np.dot(p_k.T,np.dot(A,p_k)))
		xk_plusone=x_k+alpha_k*p_k
		r_kplusone=r_k+alpha_k*np.dot(A,p_k)
		beta_kplusone=np.dot(r_kplusone.T,r_kplusone)/np.dot(r_k.T,r_k)
		p_kplusone=(-1)*r_kplusone+beta_kplusone*p_k
		r_k=r_kplusone
		p_k=p_kplusone
		x_k=xk_plusone
		if(np.linalg.norm(np.dot(A, x_k) - b) / np.linalg.norm(b))<1e-6:
			break
	x_true=np.dot(np.linalg.inv(np.dot(A.T,A)),np.dot(A.T,b))
	print('---done!---')
	return x_true