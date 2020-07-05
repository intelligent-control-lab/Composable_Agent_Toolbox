#Jaskaran
import pdb
import numpy as np
class NaiveEstimator(object):
    def __init__(self, spec, model):
        self.spec = spec
        self.posterior_state     = spec["init_x"]
        self.posterior_state_cov = spec["init_variance"]
        self._Rww                = spec["Rww"]
        self._Rvv                = spec["Rvv"]
        self.model               = model
        self.sp                  = spec["speed_factor"]
        self.dt                  = spec["time_sample"]
        self.pred_state          = None
        self.cache               = {}

    def estimate(self, u,sensor_data):
        est_data = {}
        # print(sensor_data)
        for sensor, measurement in sensor_data.items():
            est_data[sensor+"_est"] = measurement
        est_param = {
            "state_est": sensor_data["state_sensor"]["state"]
        }
        # print(est_param["state_est"])
        return est_data, est_param


    def estimate_EKF(self, u,sensor_data):
        est_data = {}
        for sensor, measurement in sensor_data.items():
            est_data[sensor+"_est"] = measurement
        est_param = {
        }
        priori_state             = np.ravel(self.model.disc_model_lam([self.posterior_state, u, [2]]))               
        A                        = np.array([[self.dt/2,0,1 + (self.sp*self.dt**2)/2,0],[0,self.dt/2,0,1 + (self.sp*self.dt**2)/2],[1,0,self.sp*self.dt,0],[0,1,0,self.sp*self.dt]])
        priori_cov               = np.matmul(np.matmul(A,self.posterior_state_cov),np.transpose(A)) + self._Rww      ;   
        n                        = len(self.posterior_state)                                                         ;
        I                        = np.identity(n)                                                               ;
        H                        = I 
        # H                        = self.linearization_measurement_function(priori_state)                        ; # need to call dynamics of the paritcular robot model
        z_meas                   = np.ravel(sensor_data["state_sensor"]["state"])
        innovation               = z_meas - np.ravel(priori_state)
        S                        = np.matmul(np.matmul(H,priori_cov),np.transpose(H)) + self._Rvv               ;
        Sinv                     = np.linalg.inv(S)                                                             ;
        K                        = np.matmul(np.matmul(priori_cov,np.transpose(H)),Sinv)                        ;
        self.posterior_state     = priori_state + np.matmul(K,innovation);                                       
        self.posterior_state_cov = np.matmul(I-(np.matmul(K,H)),priori_cov)                                     ;
        est_param["state_est"]   = np.ravel(self.posterior_state)
        return est_data, est_param
    