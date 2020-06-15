import numpy as np

def jac_num(ineq, x, eps=1e-5):
	'''
	compoute the jaccobian for a given function 
	used for computing first-order gradient of distance function 
	'''
	y = ineq(*x)
	grad = np.zeros(len(y), len(x))
	xp = x
	for i in range(len(x)):
		xp[i] = x[i] + eps/2
		yhi = ineq(*xp)
		xp[i] = x[i] - eps/2
		ylo = ineq(*xp)
		grad[:,i] = (yhi - ylo) / eps
		xp[i] = x[i]
	return grad