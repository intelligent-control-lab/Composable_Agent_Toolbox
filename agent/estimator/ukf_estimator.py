#Jaskaran
import pdb
import numpy as np
from scipy.linalg import sqrtm

class UKFEstimator(object):
    def __init__(self, spec, model):
        self.spec = spec
        self.posterior_state     = spec["init_x"]
        self.posterior_state_cov = spec["init_variance"]
        self._Rww                = spec["Rww"]
        self._Rvv                = spec["Rvv"]
        self._alpha_ukf          = spec["alpha_ukf"]
        self._kappa_ukf          = spec["kappa_ukf"]
        self._beta_ukf           = spec["beta_ukf"]
        self.kp                  = spec["kp"]
        self.kv                  = spec["kv"]
        self.model               = model
        self.dt                  = spec["time_sample"]
        self.pred_state          = None
        self.cache               = {}


    def estimate(self, u,sensor_data):
        # This is the main estimate function, it takes the state and state covariance and measuremnets
        # and returns state and covariance at the next time step
        
        other_agent_state = self.posterior_state


        est_data = {}
        for sensor, measurement in sensor_data.items():
            est_data[sensor+"_est"] = measurement
        est_param = {}

        for name in sensor_data["communication_sensor"].keys():
            st = sensor_data["communication_sensor"][name]
            for things in st.keys():
                other_agent_state = np.ravel(st[things])


        X                        = self.get_sigma_points(self.posterior_state,self.posterior_state_cov)

        N                        = len(self.posterior_state)
        priori_pos_sigma         = np.zeros((N,(2*N)+1))
        for i in range((2*N)+1):
            v                     = X[:,i]
            z                     = (self.model.disc_model_lam([v, u, [2]]))
            priori_pos_sigma[:,i] = z.ravel().reshape((N,))
            

        z_meas                   = np.vstack(np.ravel(sensor_data["state_sensor"]["state"]))
        weighted_pos_mean        = self.compute_weighted_mean(priori_pos_sigma)
        weighted_pos_covariance  = self.compute_weighted_covariance(weighted_pos_mean,priori_pos_sigma,self._Rww)
        X                        = self.get_sigma_points(weighted_pos_mean,weighted_pos_covariance);
        Z                        = X ; 
        weighted_meas_mean       = self.compute_weighted_mean(Z);
        weighted_meas_covariance = self.compute_weighted_covariance(weighted_meas_mean,Z,self._Rvv);
        F                        = self.compute_other_covariance(weighted_pos_mean,X,weighted_meas_mean,Z);
        K                        = np.matmul(F,np.linalg.inv(weighted_meas_covariance));
        innovation               = z_meas - weighted_meas_mean.reshape((N,1))
        correction               = np.matmul(K,innovation);
        self.posterior_state     = weighted_pos_mean.reshape((N,1)) + correction          ;       
        self.posterior_state_cov = weighted_pos_covariance - np.matmul(np.matmul(K,weighted_meas_covariance),K.transpose())    ;
        

        est_param["ego_state_est"]   = np.vstack(np.ravel(self.posterior_state))
        est_param["other_state_est"] = np.vstack(other_agent_state)



        return est_data, est_param


    def get_sigma_points(self,mu,E):
        # This function gets 2N+1 sigma points centered around the mean mu with variance in multiples of E
        # They are collected in X
        N                       = len(mu)
        X                       = np.zeros((N,(2*N)+1))
        X[:,0]                  = mu.ravel()
        lambda_ukf              = ((self._alpha_ukf**2)*(N + self._kappa_ukf))-N;
        Z                       = sqrtm((N + lambda_ukf)*E);
        
        for i in range(N):
            X[:,i+1]            = mu.ravel() + Z[:,i].ravel()
            
        for i in range(N):
            X[:,i+1+N]          = mu.ravel() - Z[:,i].ravel()
        
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
    