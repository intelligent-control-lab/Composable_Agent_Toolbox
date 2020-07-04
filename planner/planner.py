# import math
import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from .src.utils import *
import ipdb

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

    def plan(self, ineq, eq, agent_state):
        pass


    # @property
    # def inequality_cons(self):
    #     return self._func

    # @property
    # def equality_cons(self):
    #     return self._func  
    

class OptimizationBasedPlanner(Planner):
    def __init__(self, spec, model):
        super().__init__(spec, model)
        self.spec = spec
        self.model = model
        self.replanning_cycle = spec['replanning_cycle']
        self.horizon = spec['horizon']
        self.cache = {}

    def planning(self, dt, goal, agent_start_state):
        target = goal[0:2,0]
        state = agent_start_state['state_sensor_est']['state'][0:2,0]
        ref_traj = self.plan(self.ineq, self.eq, state, target)
        return ref_traj

    def re_planning(self, dt, goal, agent_state):
        agent_next_state = agent_state
        ref_traj = plan(self.ineq, self.eq, agent_next_state, goal)
        return agent_next_state, ref_traj

    def reference_traj_generate(self, agent_state, goal):
        '''
        generate the reference trajectory 
        '''
        # transform to numpy array 
        agent_state = np.array(agent_state)
        goal = np.array(goal)
        traj = np.zeros((self.spec['horizon'], self.spec['dim']))
        for i in range(self.spec['horizon']):
            traj[i, :] = agent_state + i / self.spec['horizon'] * (goal - agent_state)

        return traj

    def plan(self, ineq, eq, agent_state, goal):
        ref_traj = self.reference_traj_generate(agent_state, goal)
        # ipdb.set_trace()
        traj = self.CFS(x_ref=ref_traj)
        return traj 

    def ineq(self, x):
        '''
        inequality constraints. 
        constraints: ineq(x) > 0
        '''
        # norm distance restriction
        obs_p = self.spec['obsp']
        obs_r = self.spec['obsr']
        obs_p = np.array(obs_p)
        obs_r = np.array(obs_r)
        
        # flatten the input x 
        x = x.flatten()
        dist = np.linalg.norm(x - obs_p) - obs_r - 0.5
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
        cq = [1,0,0], 
        cs = [0,10,10], 
        minimal_dis = 0, 
        ts = 1, 
        maxIter = 30,
        stop_eps = 1e-3
    ):
        # without obstacle, then collision free
        n_ob = self.spec['n_ob']
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
        I = np.identity(h * dimension)
        # velocity terms
        Velocity = np.zeros(((h - 1) * dimension, h * dimension))
        for i in range(len(Velocity)):
            Velocity[i][i] = 1.0
            Velocity[i][i + dimension] = -1.0
        Velocity /= ts
        # acceleration terms 
        Acceleration = np.zeros(((h - 2) * dimension, h * dimension))
        for i in range(len(Acceleration)):
            Acceleration[i][i] = 1.0
            Acceleration[i][i + dimension] = -2.0
            Acceleration[i][i + dimension + dimension] = 1.0
        Acceleration /= (ts * ts)

        # objective 
        Q = cq[0] * I + cq[1] * np.dot(np.transpose(Velocity), Velocity) + cq[2] * np.dot(np.transpose(Acceleration), Acceleration)
        S = cs[0] * I + cs[1] * np.dot(np.transpose(Velocity), Velocity) + cs[2] * np.dot(np.transpose(Acceleration), Acceleration)

        # weight terms 

        # quadratic term
        H =  Q + S 
        # linear term
        f = -2 * np.dot(Q, x_origin)


        b = np.ones((h * n_ob, 1)) * (-minimal_dis)

        H = matrix(H,(len(H),len(H[0])),'d')
        f = matrix(f,(len(f), 1),'d')
        b = matrix(b,(len(b),1),'d')

        # reference trajctory cost 
        J0 =  np.dot(np.transpose(x_rs - x_origin), np.dot(Q, (x_rs - x_origin))) + np.dot(np.transpose(x_rs), np.dot(S, x_rs))
        J = float('inf')
        dlt = float('inf')
        cnt = 0

        # equality constraints 
        # start pos and end pos remain unchanged 
        Aeq = np.zeros((dimension*2, len(x_rs)))
        for i in range(dimension):
            Aeq[i,i] = 1
            Aeq[dimension*2-i-1, len(x_rs)-i-1] = 1
        
        beq = np.zeros((dimension*2, 1))
        beq[0:dimension,0] = x_rs[0:dimension,0]
        beq[dimension:dimension*2, 0] = x_rs[dimension*(h-1): dimension*h, 0] 
        # transform to convex optimization matrix 
        Aeq = matrix(Aeq,(len(Aeq),len(Aeq[0])),'d')
        beq = matrix(beq,(len(beq),1),'d')

        # main CFS loop
        while dlt > stop_eps:
            cnt += 1
            print(f"the iteration: {cnt}")
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
                l = -1 * ref_grad

                # update 
                Sstack = vstack_wrapper(Sstack, s)
                l_tmp = np.zeros((1, len(x_rs)))
                l_tmp[:,i*dimension:(i+1)*dimension] = l
                Lstack = vstack_wrapper(Lstack, l_tmp)

            Lstack = matrix(Lstack,(len(Lstack),len(Lstack[0])),'d')
            Sstack = matrix(Sstack,(len(Sstack),1),'d')

            # QP solver 
            sol = solvers.qp(H, f, Lstack, Sstack, Aeq, beq)
            x_ts = sol['x']
            x_ts = np.reshape(x_ts, (len(x_rs),1))

            # ipdb.set_trace()
            J = np.dot(np.transpose(x_ts - x_origin), np.dot(Q, (x_ts - x_origin))) + np.dot(np.transpose(x_ts), np.dot(S, x_ts))
            dlt = min(abs(J - J0), np.linalg.norm(x_ts - x_rs))
            J0 = J
            x_rs = x_ts
            if cnt >= maxIter:
                break

            # traj = x_rs[: h * dimension].reshape(h, dimension)
            # plt.clf()
            # plt.plot(traj[:,0],traj[:,1])
            # plt.pause(0.05)

        # return the reference trajectory        
        x_rs = x_rs[: h * dimension]
        plt.show()
        return x_rs.reshape(h, dimension)



class SamplingBasedPlanner(Planner):
    def __init__(self, spec, model):
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



if __name__ == '__main__':
    from src.configs import add_planner_args
    from pprint import pprint
    import argparse
    
    '''
    test the planner class
    test the convex feasible set algorithm
    '''
    parser = argparse.ArgumentParser(description='Planning Parameters Setting')
    parser = add_planner_args(parser)
    args, unknown = parser.parse_known_args()
    args = vars(args)

    experiment_settings = load_experiment_settings(args['experiment_settings'])
    args.update(experiment_settings)

    pprint(args)

    # models 
    model = 1 # the place holder 

    CFS  = OptimizationBasedPlanner(args, model)
    traj = CFS.planning(args['goal'], args['state'])
    print(traj)
    plt.plot(traj[:,0],traj[:,1])
    plt.show()





