import numpy as np
from scipy.linalg import sqrtm
import math
import pdb;
global g,L,dt
g = 9.81
L = 1 
dt = 0.01;

class LQR_Control:

	def __init__(self,controller_params,model_params):
		self.name 				 = 'LQR'
		self.params 			 = controller_params
		self._A    			 	 = model_params['A'] # passing A as a dictionary
		self._B    			 	 = model_params['B'] # passing B as a dictionary
		self._Q    			 	 = controller_params['Q'] # passing Q as a dictionary
		self._R    			 	 = controller_params['R'] # passing R as a dictionary
		

	def param_tuning(self,params):
		self._Q 				 = params['Q']
		self._R 				 = params['R']
		return True


	def compute_LQR_gain(self):
		_A 						= self._A;
		_B 						= self._B;
		_Q						= self._Q;
		_R 						= self._R;
		N,C 					= _A.shape
		H11 					= _A;
		H12						= -np.matmul(np.matmul(_B,np.linalg.inv(_R)),np.transpose(_B))
		H21 					= -_Q;
		H22 					= -np.transpose(_A);
		H 						= np.block([[H11, H12], [H21, H22]])
		w, v 					= np.linalg.eig(H)
		X 						= np.zeros((2*N,N),dtype=complex);
		count 					= 0 ;

		for i in range(2*N):
			if w[i]<0:
				X[:,count] = v[:,i];
				count = count + 1

			
		X1 		= X[0:N,:];
		X2 		= X[N:(2*N),:];
		P    	= np.real(np.matmul(X2,np.linalg.inv(X1)))
		K 		= np.matmul(np.matmul(np.linalg.inv(_R),np.transpose(_B)),P)
	
		return K

	def reset(self):
		return True

if __name__ == '__main__':
	n      		  		= 3;
	m 			  		= 2 ; 
	Q    		  		= 1*np.identity(n)
	R    		  		= 2*np.identity(m)
	A     		  		= np.array([[1.,2.,3.],[4.,0.,6.],[17.,8.,9.]]) ; 
	B     		  		= np.array([[0.0,2.0],[1.0,1.0],[0.0,3.0]]) ; 
	model_params 		= {'A':A,'B':B}
	controller_params 	= {'Q':Q,'R':R}
	LQR1 				= LQR_Control(controller_params,model_params)
	K 					= LQR1.compute_LQR_gain()
	print(K)
	