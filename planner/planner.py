import math
import numpy as np
from .util import *
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

solvers.options['show_progress'] = False

'''Weiye: motion planning'''
class Planner(object):
    def __init__(self, spec, model):
        self.spec = spec
        self.model = model
        self.cache = {}
    def planning(self, dt, goal, agent_state):
        agent_next_state = agent_state
        return agent_next_state
    def re_planning(self, dt, goal, agent_state):
        agent_next_state = agent_state
        return agent_next_state

    @abstractmethod
    def plan(self, ineq, eq, agent_state):
        pass


    @property
    def inequality_cons(self):
        return self._func

    @property
    def equality_cons(self):
        return self._func  
    

class OptimizationBasedPlanner(Planner)
    def __init__(self, spec, model)
        super().__init__(self, spec, model)
            self.spec = spec
            self.model = model
            self.cache = {}

    def planning(self, dt, goal, agent_start_state):
        ref_traj = plan(self.ineq, self.eq, agent_start_state, goal)
        return agent_next_state, ref_traj

    def re_planning(self, dt, goal, agent_state):
        agent_next_state = agent_state
        ref_traj = plan(self.ineq, self.eq, agent_next_state, goal)
        return agent_next_state, ref_traj

    def reference_traj_generate(self, agent_state, goal):
        '''
        generate the reference trajectory 
        '''
        traj = np.zeros(self.spec.horizon, self.spec.dim)
        for i in range(self.spec.horizon):
            traj[i, :] = agent_state + i / self.spec.horizon * (goal - agent_state)

        return traj

    def plan(self, ineq, eq, agent_state, goal):
        ref_traj = reference_traj_generate(agent_state, goal)
        traj = CFS(x_ref=ref_traj)
        return traj 

    def ineq(self, x):
        '''
        inequality constraints. 
        constraints: ineq(x) > 0
        '''
        # an simple example 
        obs_p = self.spec.obsp
        obs_r = self.spec.obsr
        dist = np.norm(x - obs_p) - obs_r
        return dist

    def eq(self, x):
        '''
        equality constraints. 
        constraints: eq(x) = 0
        '''
        pass

    def CFS(
        self, 
        x_ref,
        cq = [1,1,1], 
        cs = [1,1,1], 
        minimal_dis = 0, 
        ts = 1, 
        maxIter = 10,
        stop_eps = 1e-3
    ):
        # without obstacle, then collision free
        n_ob = self.spec.n_ob
        if n_ob == 0:
            return x_ref

        # has obstacle, the normal CFS procedure 
        x_rs = np.array(x_ref)

        # planning parameters 
        h = x_rs.shape[0]    
        dimension = x_rs.shape[1] #

        # flatten the trajectory to one dimension
        # flatten to one dimension for applying qp, in the form of x0,y0,x1,y1,...
        x_rs = np.reshape(x_rs, (x_rs.size, 1)) 
        x_origin = x_rs

        # objective terms 
        # identity
        I = np.identity(h * dimension + 1)
        # velocity terms
        Velocity = np.zeros(((h - 1) * dimension, h * dimension + 1))
        for i in range(len(Velocity)):
            Velocity[i][i] = 1.0
            Velocity[i][i + dimension] = -1.0
        Velocity /= ts
        # acceleration terms 
        Acceleration = np.zeros(((h - 2) * dimension, h * dimension + 1))
        for i in range(len(Acceleration)):
            Acceleration[i][i] = 1.0
            Acceleration[i][i + dimension] = -2.0
            Acceleration[i][i + dimension + dimension] = 1.0
        Acceleration /= (ts * ts)

        # objective 
        Q = cq[0] * I + cq[1] * np.dot(np.transpose(Velocity), Velocity) + cq[2] * np.dot(np.transpose(Acceleration), Acceleration)
        S = cs[0] * I + cs[1] * np.dot(np.transpose(Velocity), Velocity) + cs[2] * np.dot(np.transpose(Acceleration), Acceleration)

        # weight terms 
        w1 = 1
        w2 = 1

        # quadratic term
        H = w1 * Q + w2 * S 
        # linear term
        f = -2 * w1 * np.dot(Q, x_origin)


        b = np.ones((h * n_ob, 1)) * (-minimal_dis)

        H = matrix(H,(len(H),len(H[0])),'d')
        f = matrix(f,(len(f), 1),'d')
        b = matrix(b,(len(b),1),'d')

        # reference trajctory cost 
        J0 = w1 * np.dot(np.transpose(x_rs - x_origin), np.dot(Q, (x_rs - x_origin))) + w2 * np.dot(np.transpose(x_rs), np.dot(S, x_rs))
        J = float('inf')
        dlt = float('inf')
        cnt = 0

        # equality constraints 
        # start pos and end pos remain unchanged 
        Aeq = np.zeros(dimension*2, len(x_rs))
        for i in range(dimension):
            Aeq[i,i] = 1
            Aeq[dimension*2-i+1, len(x_rs)-i+1] = 1
        beq = np.zeros(dimension*2, 1)
        beq[0:dimension,0] = x_rs[0:dimension,0]
        beq[dimension:dimension*2, 0] = x_rs[dimension*(h-1): dimension*h, 0] 


        # main CFS loop
        while dlt > stop_eps:
            cnt += 1

            Lstack, Sstack = [], []
            # inequality constraints 
            # l * x <= s
            Constraint = np.zeros((h * n_ob, len(x_rs)))
            for i in range(h):
                # get reference pos at time step i
                x_r = x_rs[i * dimension : (i + 1) * dimension] 

                # get inequality value (distance)
                dist = self.ineq(x_r)

                # get gradient 
                ref_grad = jac_num(self.ineq, x_r)

                # compute
                s = dist - np.dot(ref_grad, x_r)
                l = -1 * ref_grad.transpoes()

                # update 
                Sstack = np.vstack(Sstack, s)
                l_tmp = np.zeros(1, len(x_rs))
                l_tmp[(i-1)*dimension:i*dimension] = l
                Lstack = np.vstack(Lstack, l_tmp)

            # transform to convex optimization matrix 
            Aeq = matrix(Aeq,(len(Aeq),len(Aeq[0])),'d')
            beq = matrix(beq,(len(beq),1),'d')
            Lstack = matrix(Lstack,(len(Lstack),len(Lstack[0])),'d')
            Sstack = matrix(Sstack,(len(Sstack),1),'d')

            # QP solver 
            sol = solvers.qp(H, f, Lstack, Sstack, Aeq, beq)
            x_ts = sol['x']
            x_ts = np.reshape(x_ts, len(x_rs))

            J = w1 * np.dot(np.transpose(x_ts - x_origin), np.dot(Q, (x_ts - x_origin))) + w2 * np.dot(np.transpose(x_ts), np.dot(S, x_ts))
            dlt = min(abs(J - J0), np.linalg.norm(x_ts - x_rs))
            J0 = J
            x_rs = x_ts
            if cnt >= maxIter:
                break


        # return the reference trajectory        
        x_rs = x_rs[: h * dimension]
        return x_rs.reshape(h, dimension)



class SamplingBasedPlanner(Planner)
    def __init__(self, spec, model)
        super().__init__(self, spec, model)
            self.spec = spec
            self.model = model
            self.cache = {}

    def planning(self, dt, goal, agent_state):
        agent_next_state = agent_state
        return agent_next_state

    def re_planning(self, dt, goal, agent_state):
        agent_next_state = agent_state
        return agent_next_state

    def plan(self, ineq, eq, agent_state):
        pass



if __name__ == "__main__":
    '''
    test the planner class
    test the convex feasible set algorithm
    '''

    # vh_w = 1.2
    # vh_l = 2.8
    # obstacles = [[0,3.0]]
    # refTraj = np.array([[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,9]])

    # plt.subplot(1,3,1)
    # plt.title('CFS')
    # traj1 = CFS(refTraj[0], refTraj, obstacles, cq = [0.1,0,0], cs = [0.1,0,1], minimal_dis = 0.1, maxIter = 10, SCCFS = False)

