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

	def reset(self):
		return True



