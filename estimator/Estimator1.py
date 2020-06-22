import math
import numpy as np
from abc import ABCMeta, abstractmethod


class Estimator_Base():
	__metaclass__ = ABCMeta

	estimator_name = ''
	estimator_type = '' # EKF,UKF,RLS etc.

	@abstractmethod
	def __init__(self):
		pass
	
	@abstractmethod
	def estimate(self):
		'''
		This function calculates the next state  each time step
		'''
		pass

	@abstractmethod
	def reset(self):
		'''
		return TRUE for success, FALSE for failure
		'''
		pass

	def get_name(self):
		return self.estimator_name

	def get_type(self):
		return self.estimator_type


class Zero_Estimator(Estimator_Base):
	'''
	This is a blank estimator class.
	The Zero_Estimator always returns 0 as estimator output.
	'''
	def __init__(self, spec, model):
		self.name = 'Zero'
		self.type = 'None'
		self.spec = spec
		self.model = model

	def reset(self):
		return True

	def estimate(self, x,covariance,z):
		return np.zeros(self.model.x_shape)


class Estimator(Estimator_Base):
	'''
    This is the unified estimator class used in Agent Class.
    Estimator specs should be given in the following format:
    {'Extended Kalman Filter': EFK, 'Unscented Kalman Filter': 'UKF'}
    The estimator key can be ignored if it's not used, it will be filled with an zero estimator automatically.
    
    E.g. EKF Estimator specs:
    {"Extended Kalman filter": "EKF", "params": {"Rww": [1,2], "Rvv": [0, 0]}}
    ''' 
	def __init__(self,spec,model):
		self.spec  = spec
		self.model = model

		if 'EKF' in self.spec:
			self.EKF = globals()[self.spec['EKF']](self.spec['params'],self.model) 
			# this line will take input from spec, check reflection in python. it takes input from spec and 
			# translates a string used in spec and creates a class object deepnds on string
			# example : If I define a feedback controller, with PID, when I change PID to something else like adatpive controller
			# it will automatically create an ARC class object.

		else:
			self.EKF = Zero_Estimator(self.spec,self.model)



class Unscented_Kalman_Filter(Estimator_Base):

	def __init__(self,estimator_params,model):
		self.name 				 = 'UKF'
		self.params 			 = estimator_params
		self.model  			 = model
		self._Rww    			 = estimator_params['Rww'] # passing covariances as a dictionary
		self._Rvv    			 = estimator_params['Rvv'] # passing covariances as a dictionary
		self._alpha_ukf 		 = estimator_params['alpha_ukf']
		self._kappa_ukf 		 = estimator_params['kappa_ukf']
		self._beta_ukf 		 	 = estimator_params['beta_ukf']

	def param_tuning(self,params):
		self._Rww 				 = params['Rww']
		self._Rvv 				 = params['Rvv']
		return True


	def estimate(self,posterior_pos,posterior_state_cov,z_meas):
		X						 = self.get_sigma_points(posterior_pos,posterior_state_cov)							
		priori_pos_sigma		 = self.forward_propagate_dynamics(X)													
		weighted_pos_mean		 = self.compute_weighted_mean(priori_pos_sigma)
		weighted_pos_covariance  = self.compute_weighted_covariance(weighted_pos_mean,priori_pos_sigma,self._Rww)
		X                        = self.get_sigma_points(weighted_pos_mean,weighted_pos_covariance);
		Z                        = self.measurement_function(X) ; 
		weighted_meas_mean       = self.compute_weighted_mean(Z);
		weighted_meas_covariance = self.compute_weighted_covariance(weighted_meas_mean,Z,Rvv);
		F                        = self.compute_other_covariance(weighted_pos_mean,X,weighted_meas_mean,Z);
		K                        = np.matmul(F,np.linalg.inv(weighted_meas_covariance));
		posterior_pos            = weighted_pos_mean + np.matmul(K,z_meas-weighted_meas_mean)          ; 
		P_post                   = weighted_pos_covariance - np.matmul(np.matmul(K,weighted_meas_covariance),K.transpose())    ;


		return posterior_pos,P_post


	def get_sigma_points(self,mu,E):
		N 						= len(mu)
		X						= np.zeros((N,(2*N)+1))
		X[:,0]					= mu
		lambda_ukf				= ((self._alpha_ukf**2)*(N + self._kappa_ukf))-N;
		Z 						= sqrtm((N + lambda_ukf)*E);
		
		for i in range(N):
			X[:,i+1] = mu + Z[:,i]
			
		for i in range(N):
			X[:,i+1+N] = mu - Z[:,i]
		
		return X


	def compute_weighted_mean(self,X):
		N,C						= X.shape
		lambda_ukf				= ((self._alpha_ukf**2)*(N + self._kappa_ukf))-N;
		w 						= (0.5/(N+lambda_ukf))*np.ones((C,))
		w[0]	 				= lambda_ukf/(N+lambda_ukf);
		mu 						= np.zeros((N,))
		
		for i in range(C):
			mu = mu + w[i]*X[:,i]	

		return mu


	def compute_weighted_covariance(self,mu,X,F):
		N,C						= X.shape
		lambda_ukf				= ((self._alpha_ukf**2)*(N + self._kappa_ukf))-N;
		w 						= (0.5/(N+lambda_ukf))*np.ones((C,))
		w[0]	 				= (lambda_ukf/(N+lambda_ukf)) + (1-(self._alpha_ukf**2) + self._beta_ukf);
		P 						= np.zeros((N,N))
		for i in range(C):
			Z = X[:,i] - mu
			P = P + w[i]*np.outer(Z,Z)
		P = P + F

		return P


	def compute_other_covariance(self,weighted_pos_mean,X,weighted_meas_mean,Z):
		Nx,N						= X.shape
		Nz,rr 						= Z.shape
		P 	 	 					= np.zeros((Nx,Nz))
		lambda_ukf					= ((self._alpha_ukf**2)*(N + self._kappa_ukf))-N;
		w 							= (0.5/(N+lambda_ukf))*np.ones((N,))
		w[0]	 					= (lambda_ukf/(N+lambda_ukf)) + (1-(self._alpha_ukf**2) + self._beta_ukf);

		for i in range(N):
			G = X[:,i] - weighted_pos_mean
			H = Z[:,i] - weighted_meas_mean
			W = w[i]*np.outer(G,H)
			P = P + W

		return P


	def reset(self):
		return True


# put underscore to internal variables
class Extended_Kalman_Filter(Estimator_Base):

	def __init__(self,estimator_params,model):
		self.name 				 = 'EKF'
		self.params 			 = estimator_params
		self.model  			 = model
		self._Rww    			 = estimator_params['Rww']
		self._Rvv    			 = estimator_params['Rvv']

	def param_tuning(self,params):
		self._Rww 				 = params['Rww']
		self._Rvv 				 = params['Rvv']
		return True

	def estimate(self,posterior_pos,posterior_state_cov,z_meas):
		# check Chase's code, see which method gives next state
		priori_pos               = self.forward_propagate_dynamics(posterior_pos)      							; # need to call dynamics of the paritcular robot model
		A                        = self.linearization_forward_dynamics(posterior_pos)  							; # need to call linearization of dynamics of the paritcular robot model
		priori_cov               = np.matmul(np.matmul(A,posterior_state_cov),np.transpose(A)) + self._Rww      ; # P- = APA' + Q 
		n                        = len(posterior_pos)                                           				; # get size of EKF state
		I                        = np.identity(n)                                               				; # initialize identity
		H                        = self.linearization_measurement_function(priori_pos)   						; # linearization of robot's measurement function
		innovation               = z_meas - self.measurement_function(priori_pos)        						; # calculate innovation, needs measurement function
		S                        = np.matmul(np.matmul(H,priori_cov),np.transpose(H)) + self._Rvv  				; # S = HP-H' + R
		Sinv                     = np.linalg.inv(S)																; # Calculate Sinverse
		K                        = np.matmul(np.matmul(priori_cov,np.transpose(H)),Sinv)                  		; # Calculate Kalman Gain
		next_posterior_state     = priori_pos + np.matmul(K,innovation)                         				; # Get next pos estimate
		next_posterior_state_cov = np.matmul(I-(np.matmul(K,H)),priori_cov)                     				; # Get next pos covaraince, this is posteriori covariance

		return next_posterior_state,next_posterior_state_cov

	def reset(self):
		return True



