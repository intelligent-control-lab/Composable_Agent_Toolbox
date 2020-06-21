import numpy as np
import math
import pdb;
global g,L,dt
g = 9.81
L = 1 
dt = 0.01;

class Extended_Kalman_Filter:

	def __init__(self,estimator_params,model):
		self.name 				 = 'EKF'
		self.params 			 = estimator_params
		self.model  			 = model
		self._Rww    			 = estimator_params['Rww'] # passing covariances as a dictionary
		self._Rvv    			 = estimator_params['Rvv'] # passing covariances as a dictionary

	def param_tuning(self,params):
		self._Rww 				 = params['Rww']
		self._Rvv 				 = params['Rvv']
		return True

	def estimate(self,posterior_pos,posterior_state_cov,z_meas):
		# check Chase's code, see which method gives next state
		priori_pos               = self.forward_propagate_dynamics(posterior_pos)      							; # need to call dynamics of the paritcular robot model
		A                        = self.linearization_forward_dynamics(posterior_pos)  							; # need to call dynamics of the paritcular robot model
		priori_cov               = np.matmul(np.matmul(A,posterior_state_cov),np.transpose(A)) + self._Rww      ;   
		n                        = len(posterior_pos)                                           				;
		I                        = np.identity(n)                                               				;
		H                        = self.linearization_measurement_function(priori_pos)   						; # need to call dynamics of the paritcular robot model
		innovation               = z_meas - self.measurement_function(priori_pos)        						; # need to call dynamics of the paritcular robot model
		S                        = np.matmul(np.matmul(H,priori_cov),np.transpose(H)) + self._Rvv  				;
		Sinv                     = np.linalg.inv(S)																;
		K                        = np.matmul(np.matmul(priori_cov,np.transpose(H)),Sinv)                  		;
		next_posterior_state     = priori_pos + np.matmul(K,innovation)                         				;
		next_posterior_state_cov = np.matmul(I-(np.matmul(K,H)),priori_cov)                     				;

		return next_posterior_state,next_posterior_state_cov

	def forward_propagate_dynamics(self,z):
		global g,L,dt 
		theta 					= z[0];
		thetadot			    = z[1];
		zdot				    = np.array([thetadot,-(g/L)*np.sin(theta)])
		z_next					= z + (dt*zdot)
		return z_next

	def linearization_forward_dynamics(self,z):
		theta 					= z[0];
		thetadot			    = z[1];
		A 					    = np.array([[1,dt],[-(dt*g*np.cos(theta)/L),1]])
		return A

	def measurement_function(self,z):
		return z

	def linearization_measurement_function(self,z):
		n  = len(z)
		H  = np.identity(n)
		return H

	def reset(self):
		return True

if __name__ == '__main__':
	n      		  = 2;
	Rww    		  = (0.5**2)*np.identity(n)
	Rvv    		  = (0.1**2)*np.identity(n)
	params 	 	  = {'Rww':Rww,'Rvv':Rvv}
	model  		  = {}
	EKF1   		  = Extended_Kalman_Filter(params,model)
	z      		  = np.zeros((2,))
	posterior_pos = np.array([0.4565,-0.2292])
	P_post 		  = 2*np.identity(2)
	z_meas 		  = np.array([0.5341,0.0235])
	print(posterior_pos)
	print(P_post)
	pnext,Covnext = EKF1.estimate(posterior_pos,P_post,z_meas)
	print(pnext)
	print(Covnext)