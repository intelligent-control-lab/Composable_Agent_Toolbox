#Jaskaran
import pdb
import numpy as np
from scipy.linalg import sqrtm
class RLSPredictor(object):
    def __init__(self, spec, model):
        self.spec                = spec
        self.posterior_state     = spec["init_x"]
        self.posterior_state_cov = spec["init_variance"]
        self.other_agent_goal    = spec["other_goal"]
        self._Rww                = spec["Rww"]
        self._Rvv                = spec["Rvv"]
        self.kp                  = spec["kp"]
        self.kv                  = spec["kv"]
        self.model               = model
        self.dt                  = spec["time_sample"]
        self.epsilon             = np.zeros((4,1))
        self.C                   = np.zeros((4,12))
        self.C[0:5,0:4]          = np.identity(4)
        self.F                   = np.identity(12)
        self.alpha               = 0.8
        self.lamb                = 0.98
        self.pred_state          = None
        self.cache               = {}

    def predict(self, u,sensor_data):
        other_agent_state = np.zeros((4,1))

        est_data = {}
        for sensor, measurement in sensor_data.items():
            est_data[sensor+"_est"] = measurement
        est_param = {}

        for name in sensor_data["communication_sensor"].keys():
            st = sensor_data["communication_sensor"][name]
            for things in st.keys():
                other_agent_state   = np.ravel(st[things])

        z_meas                      = np.vstack(np.ravel(sensor_data["state_sensor"]["state"]))
        other_agent_measurement     = np.vstack(np.ravel(other_agent_state))
        other_agent_posterior_state = np.vstack(np.ravel(other_agent_state))
        ego_priori_state            = np.vstack(self.model.disc_model_lam([self.posterior_state, u, [2]]))               
        ucl                         = np.append(z_meas,self.other_agent_goal)
        psi                         = np.append(other_agent_posterior_state,ucl)
        psi                         = np.vstack(np.ravel(psi))
        other_agent_priori_state    = np.vstack(np.ravel(np.matmul(self.C,psi)))
        other_agent_posterior_state = (1-self.alpha)*other_agent_priori_state + (self.alpha*other_agent_measurement)
        self.C                      = self.C + np.matmul(self.epsilon,np.matmul(np.transpose(psi),self.F))
        self.epsilon                = other_agent_posterior_state - other_agent_priori_state
        expression                  = self.lamb + (np.matmul(np.transpose(psi),np.matmul(self.F,psi)))
        self.F                      = (self.F - (np.matmul(self.F,np.matmul(np.matmul(psi,np.transpose(psi)),self.F)))/expression)/self.lamb
        A                           = np.array([[1.0,0.0,self.dt,0.0],[0.0,1.0,0.0,self.dt],[ -(self.dt*self.kp)/2,0.0, 1.0 - (self.dt*self.kv)/2,0.0],[0.0,-(self.dt*self.kp)/2,0.0, 1.0 - (self.dt*self.kv)/2]])
        priori_cov                  = np.matmul(np.matmul(A,self.posterior_state_cov),np.transpose(A)) + self._Rww      ;   
        n                           = len(self.posterior_state)                                                         ;
        I                           = np.identity(n)                                                               ;
        H                           = I 
        # H                        = self.linearization_measurement_function(priori_state)                        ; # need to call dynamics of the paritcular robot model
        innovation                  = z_meas - (ego_priori_state)
        S                           = np.matmul(np.matmul(H,priori_cov),np.transpose(H)) + self._Rvv               ;
        Sinv                        = np.linalg.inv(S)                                                             ;
        K                           = np.matmul(np.matmul(priori_cov,np.transpose(H)),Sinv)                        ;
        self.posterior_state        = ego_priori_state + np.matmul(K,innovation);                                       
        self.posterior_state_cov    = np.matmul(I-(np.matmul(K,H)),priori_cov)                                     ;
        est_param["ego_state_est"]  = np.vstack(np.ravel(self.posterior_state))
        est_param["other_state_est"]= np.vstack(np.ravel(other_agent_posterior_state))

        return est_data, est_param


    