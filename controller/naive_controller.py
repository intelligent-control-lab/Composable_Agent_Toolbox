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

    def control(self, dt, est_data, cartesian_goal, est_params):
        state      = est_params["state_est"]
        state_goal = self.model_inverse(est_data, est_params, cartesian_goal)
        # state_goal = np.vstack([20,20,0,0])

        e          = (state_goal - state)
        e          = np.ravel(e)
        u          = self.kp*e[0:2] + self.kv*e[2:4]
        return u
    
    def get_Ab_for_qp(rr,ze1,ze2):
        
        hval        = ((ze1-zo1)**2 + (ze2-zo2)**2) - (rr**2)
        A           = np.array([-2*(ze1-zo1), -2*(ze2-zo2)])
        b           = np.array([hval])

        return A,b

    def cbf_control(self, dt, est_data, cartesian_goal, est_params):
        ego_state    = est_params["state_est"]
        other_state  = est_params["state_est"]
        state_goal   = self.model_inverse(est_data, est_params, cartesian_goal)

 

        ze1         = ego_state
        ze2         = other_state

        [A1,b1]     = get_Ab_for_qp(4*d,ze1,ze2)
        A1          = A1.reshape((1,2)) 
        ucap        = -kp*(state_goal-ego_state)           
        P           = np.array([[2.0, 0.0],[0.0, 2.0]])                 
        q           = np.array([-2*uxcap,-2*uycap])     
        G           = A1                      
        h           = b1        
        

        ustar       = solve_qp(P,q,G,h) 
        return ustar