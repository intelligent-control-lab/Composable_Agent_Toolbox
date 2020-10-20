import numpy as np
from scipy.linalg import sqrtm
import math
import pdb;
global g,L,dt
g = 9.81
L = 1 
dt = 0.01;

class Unscented_Kalman_Filter:

	def __init__(self,estimator_params,model):
		self.name                = 'UKF'
		self.params              = estimator_params
		self.model               = model
		self._Rww                = estimator_params['Rww'] # passing covariances as a dictionary
		self._Rvv                = estimator_params['Rvv'] # passing covariances as a dictionary
		self._alpha_ukf          = estimator_params['alpha_ukf']
		self._kappa_ukf          = estimator_params['kappa_ukf']
		self._beta_ukf           = estimator_params['beta_ukf']

	def param_tuning(self,params):
		self._Rww                = params['Rww']
		self._Rvv                = params['Rvv']
		return True


	def estimate(self,posterior_pos,posterior_state_cov,z_meas):
		# This is the main estimate function, it takes the state and state covariance and measuremnets
		# and returns state and covariance at the next time step
		X                        = self.get_sigma_points(posterior_pos,posterior_state_cov)							
		priori_pos_sigma         = self.forward_propagate_dynamics(X)													
		weighted_pos_mean        = self.compute_weighted_mean(priori_pos_sigma)
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
		# This function gets 2N+1 sigma points centered around the mean mu with variance in multiples of E
		# They are collected in X
		print(mu.shape)
		N                       = len(mu)
		X                       = np.zeros((N,(2*N)+1))
		print(X.shape)
		X[:,0]                  = mu
		lambda_ukf              = ((self._alpha_ukf**2)*(N + self._kappa_ukf))-N;
		Z                       = sqrtm((N + lambda_ukf)*E);
		
		for i in range(N):
			X[:,i+1]            = mu + Z[:,i]
			
		for i in range(N):
			X[:,i+1+N]          = mu - Z[:,i]
		
		return X


	def compute_weighted_mean(self,X):
		# This function calculates the weighted mean of the sigma points X, 
		# X[0,:] is the mean
		N,C                     = X.shape
		lambda_ukf              = ((self._alpha_ukf**2)*(N + self._kappa_ukf))-N;
		w                       = (0.5/(N+lambda_ukf))*np.ones((C,))
		w[0]                    = lambda_ukf/(N+lambda_ukf);
		mu                      = np.zeros((N,))
		
		for i in range(C):
			mu                  = mu + w[i]*X[:,i]	

		return mu


	def compute_weighted_covariance(self,mu,X,F):
		N,C                     = X.shape
		lambda_ukf              = ((self._alpha_ukf**2)*(N + self._kappa_ukf))-N;
		w                       = (0.5/(N+lambda_ukf))*np.ones((C,))
		w[0]                    = (lambda_ukf/(N+lambda_ukf)) + (1-(self._alpha_ukf**2) + self._beta_ukf);
		P                       = np.zeros((N,N))
		for i in range(C):
			Z = X[:,i] - mu
			P = P + w[i]*np.outer(Z,Z)
		P = P + F

		return P


	def compute_other_covariance(self,weighted_pos_mean,X,weighted_meas_mean,Z):
		# This function computes statistical covariance based on sigma points
		Nx,N                        = X.shape
		Nz,rr                       = Z.shape
		P                           = np.zeros((Nx,Nz))
		lambda_ukf                  = ((self._alpha_ukf**2)*(N + self._kappa_ukf))-N;
		w                           = (0.5/(N+lambda_ukf))*np.ones((N,))
		w[0]                        = (lambda_ukf/(N+lambda_ukf)) + (1-(self._alpha_ukf**2) + self._beta_ukf);

		for i in range(N):
			G = X[:,i] - weighted_pos_mean
			H = Z[:,i] - weighted_meas_mean
			W = w[i]*np.outer(G,H)
			P = P + W

		return P


	def forward_propagate_dynamics(self,X):
		global g,L,dt
		N,C    = X.shape 
		z_next = X
		
		for i in range(C):
			z                       = X[:,i]
			theta                   = z[0];
			thetadot                = z[1];
			zdot                    = np.array([thetadot,-(g/L)*np.sin(theta)])
			z_next[:,i]             = z + (dt*zdot)
		
		return z_next


	def measurement_function(self,z):
		return z


	def reset(self):
		return True

if __name__ == '__main__':
	n             = 2;
	Rww           = (0.5**2)*np.identity(n)
	Rvv           = (0.1**2)*np.identity(n)
	alpha_ukf     = 1 ; 
	kappa_ukf     = 0.1 ;
	beta_ukf      = 2 ; 
	params        = {'Rww':Rww,'Rvv':Rvv,'alpha_ukf':alpha_ukf,'kappa_ukf':kappa_ukf,'beta_ukf':beta_ukf}
	model         = {}
	UKF1          = Unscented_Kalman_Filter(params,model)
	z             = np.zeros((2,))
	posterior_pos = np.array([1,2])
	P_post        = 21*np.identity(2)
	z_meas        = np.array([1.15341,2.0235])
	[x,y]         = UKF1.estimate(posterior_pos,P_post,z_meas)
	print(x)
	print(y)





	