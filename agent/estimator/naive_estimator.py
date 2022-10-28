#Jaskaran
import pdb
import numpy as np
from scipy.linalg import sqrtm
class NaiveEstimator(object):
    def __init__(self, spec, model):
        self.spec = spec
        self.posterior_state     = np.array(spec["init_x"])
        self.posterior_state_cov = np.diag(spec["init_variance"])
        self._Rww                = np.diag(spec["Rww"])
        self._Rvv                = np.diag(spec["Rvv"])
        self._alpha_ukf          = spec["alpha_ukf"]
        self._kappa_ukf          = spec["kappa_ukf"]
        self._beta_ukf           = spec["beta_ukf"]
        self.kp                  = spec["kp"]
        self.kv                  = spec["kv"]
        self.model               = model
        self.dt                  = spec["time_sample"]
        self.pred_state          = None
        self.cache               = {}

    def estimate(self, u, sensor_data):
        est_data = {}
        for sensor, measurement in sensor_data.items():
            est_data[sensor+"_est"] = measurement
        est_param = {
            "state_est":     sensor_data["state_sensor"]["state"],
            "ego_state_est": sensor_data["state_sensor"]["state"]
        }
        return est_data, est_param

    
    def direct_pass(self, sensor_data):
        return sensor_data

    