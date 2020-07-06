#Jaskaran
class NaiveEstimator(object):
    def __init__(self, spec, model):
        self.spec = spec
        self.model = model
        self.pred_state = None
        self.cache = {}

    def estimate(self, sensor_data, last_control):
        est_data = {}
        for sensor, measurement in sensor_data.items():
            est_data[sensor+"_est"] = measurement
        est_param = {
            "state_est": sensor_data["state_sensor"]["state"]
        }
        return est_data, est_param
    