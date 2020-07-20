#Jaskaran
import pdb
import numpy as np
from scipy.linalg import sqrtm
class EKFEstimator(object):
    def __init__(self, spec, model):
        self.spec = spec
        self.posterior_state     = spec["init_x"]
        self.posterior_state_cov = spec["init_variance"]
        self._Rww                = spec["Rww"]
        self._Rvv                = spec["Rvv"]
        self.kp                  = spec["kp"]
        self.kv                  = spec["kv"]
        self.model               = model
        self.dt                  = spec["time_sample"]
        self.pred_state          = None
        self.cache               = {}


    def estimate(self, u,sensor_data):
        est_data = {}
        for sensor, measurement in sensor_data.items():
            est_data[sensor+"_est"] = measurement
        est_param = {}


        priori_state             = np.vstack(self.model.disc_model_lam([self.posterior_state, u, [2]]))               
        A                        = np.array([[1.0,0.0,self.dt,0.0],[0.0,1.0,0.0,self.dt],[ -(self.dt*self.kp)/2,0.0, 1.0 - (self.dt*self.kv)/2,0.0],[0.0,-(self.dt*self.kp)/2,0.0, 1.0 - (self.dt*self.kv)/2]])
        priori_cov               = np.matmul(np.matmul(A,self.posterior_state_cov),np.transpose(A)) + self._Rww      ;   
        n                        = len(self.posterior_state)                                                         ;
        I                        = np.identity(n)                                                               ;
        H                        = I 
        # H                        = self.linearization_measurement_function(priori_state)                        ; # need to call dynamics of the paritcular robot model
        z_meas                   = np.vstack(np.ravel(sensor_data["state_sensor"]["state"]))
        innovation               = z_meas - (priori_state)

        S                        = np.matmul(np.matmul(H,priori_cov),np.transpose(H)) + self._Rvv               ;
        Sinv                     = np.linalg.inv(S)                                                             ;
        K                        = np.matmul(np.matmul(priori_cov,np.transpose(H)),Sinv)                        ;
        self.posterior_state     = priori_state + np.matmul(K,innovation);                                       
        self.posterior_state_cov = np.matmul(I-(np.matmul(K,H)),priori_cov)                                     ;
        est_param["state_est"]   = np.vstack(np.ravel(self.posterior_state))
        return est_data, est_param


    