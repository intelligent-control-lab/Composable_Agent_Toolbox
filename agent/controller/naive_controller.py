import numpy as np
# from qpsolvers import solve_qp
class NaiveController(object):
    def __init__(self, spec, model):
        self.name = 'PD'
        self.spec = spec
        self.model = model
        self.kp = spec["kp"]
        self.kv = spec["kv"]
    def model_inverse(self, est_data, est_params, cartesian_goal):
        """This function represents the inverse function in the model
        """
        state_goal = cartesian_goal
        return state_goal

    def control(self, dt, est_data, goal, est_params):
        # it seems that cartesian goal is simply the goal itself i.e. position and velocity.
        # if we are calling it cartesian, we should omit velocity.
        cartesian_goal = np.vstack(goal)
        cartesian_state = np.vstack([est_data["cartesian_sensor_est"]["pos"], est_data["cartesian_sensor_est"]["vel"]])
        state_goal = self.model_inverse(est_data, est_params, cartesian_goal)

        e          = (cartesian_goal - cartesian_state)
        e          = np.ravel(e)
        n = len(e)
        u          = self.kp*e[:n//2] + self.kv * e[n//2:n]
        return u
        
class NaiveJointController(object):
    def __init__(self, spec, model):
        self.name = 'NaiveJoint'
        self.spec = spec
        self.model = model
        self._kp = spec["kp"]
    def control(self, dt, est_data, goal, est_params):
        # simply transfer the reference joint position to the output
        cartesian_goal = np.vstack(goal)
        n = len(cartesian_goal)
        u = self._kp * cartesian_goal[:n//2]
        return u