import numpy as np
class NaiveController(object):
    def __init__(self, spec, model):
        self.name = 'PD'
        self.spec = spec
        self.model = model
        self.speed_factor = spec["speed_factor"]
    def model_inverse(self, est_data, est_params, cartesian_goal):
        """This function represents the inverse function in the model
        """
        state_goal = cartesian_goal
        return state_goal

    def control(self, dt, est_data, cartesian_goal, est_params):
        cartesian_goal = np.vstack(cartesian_goal)
        state = est_params["state_est"]
        state = np.vstack(state)
        state_goal = self.model_inverse(est_data, est_params, cartesian_goal)
        e = state_goal - state
        u = e[[0,1],:] + e[[2,3],:] * dt * self.speed_factor
        return u
