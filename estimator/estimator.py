#Jaskaran
class Estimator(object):
    def __init__(self, spec, model):
        self.spec = spec
        self.model = model
        self.pred_state = None
        self.cache = {}

    def fusion(self, sensor_data):
        fus_sensor_data = sensor_data
        return fus_sensor_data

    def estimate(self, sensor_data):
        fus_sensor_data = self.fusion(sensor_data)
        est_param = 1
        self.model.param = est_param
        est_state = fus_sensor_data
        return est_state, est_param
    