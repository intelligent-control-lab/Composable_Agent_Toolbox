import numpy as np
from qpsolvers import solve_qp
class SafeController(object):
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
        cartesian_goal = np.vstack(cartesian_goal)
        state          = est_params["ego_state_est"]
        ego_state      = np.vstack(state)
        state          = est_params["other_state_est"]
        other_state    = np.vstack(state)
        state_goal     = self.model_inverse(est_data, est_params, cartesian_goal)
        e              = (state_goal - ego_state)
        e              = np.ravel(e)
        ucap           = self.kp*e[0:2] + self.kv*e[2:4]
        Ds             = 5
        [A1,b1]        = self.get_Ab_for_qp(Ds,ego_state,other_state)
        A1             = A1.reshape((1,2))  
        P              = np.array([[2.0, 0.0],[0.0, 2.0]])                 
        q              = -2*ucap     
        G              = A1                      
        h              = b1        
 

        ustar          = solve_qp(P,q,G,h) 
        return ustar

 
    
    def get_Ab_for_qp(self,Ds,ze1,ze2):
        p1          = ze1[0:2]
        v1          = ze1[2:4]
        p2          = ze2[0:2]
        v2          = ze2[2:4]
        pij         = (p1-p2)
        vij         = (v1-v2)

        distance    = np.linalg.norm(pij)
        h1          = np.sqrt(4*(distance-Ds)) 
        h2          = ((pij[0]*vij[0])+(pij[1]*vij[1]))
        hval        = h1 + (h2/distance)
        A           = -np.ravel(pij)
        bval        = (distance*(hval**3)) + (h2/np.sqrt(distance-Ds)) + (np.linalg.norm(vij)**2) - (h2/distance)**2
        b           = np.ravel([0.5*bval])

        return A,b

    