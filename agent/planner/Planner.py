from abc import ABC, abstractmethod
from operator import concat
from matplotlib.pyplot import axis
import numpy as np
from cvxopt import matrix, solvers
from .src.utils import *

class Planner(ABC):
    def __init__(self, spec, model) -> None:
        self.spec = spec
        self.model = model
        self.replanning_cycle = spec["replanning_cycle"]
        self.horizon = spec["horizon"]
        self._state_dimension = spec["state_dimension"]
    
    @property
    def state_dimension(self):
        return self._state_dimension

    @abstractmethod
    def _plan(self, dt: float, goal: dict, est_data: dict) -> np.array:
        '''
            Implementation of planner
        '''
        pass

    def __call__(self, dt: float, goal: dict, est_data: dict) -> np.array:
        '''
            Public interface
        '''
        return self._plan(dt, goal, est_data)

class NaivePlanner(Planner):
    def __init__(self, spec, model):
        super().__init__(spec, model)

        # self.cache = {}

    def _plan(self, dt: float, goal: dict, est_data: dict) -> np.array:

        super()._plan(dt, goal, est_data)

        pos_vel = np.vstack([est_data["cartesian_sensor_est"]["pos"], est_data["cartesian_sensor_est"]["vel"]])
        traj = []
        goal_pos_vel = goal['goal']
        frac = (goal_pos_vel - pos_vel)*1./self.horizon
        for i in range(self.horizon):
            traj.append(pos_vel + frac*i)

        return np.array(traj)

    # if necessary, create new class
    # def planning_arm(self, dt, goal, est_data):
    #     pos_vel = np.vstack([est_data["cartesian_sensor_est"]["pos"], est_data["cartesian_sensor_est"]["vel"]])
    #     traj = []
    #     goal = goal['goal']
    #     frac = (goal - pos_vel)*1./self.horizon
    #     for i in range(self.horizon):
    #         traj.append(pos_vel + frac*i)
    #     return np.array(traj)

class IntegraterPlanner(Planner):

    '''
        Apply double/triple/etc. (defined by planning model) integrater
        on each state dimension
    '''

    def __init__(self, spec, model) -> None:
        super().__init__(spec, model)
    
    def _plan(self, dt: float, goal: dict, est_data: dict) -> np.array:
        super()._plan(dt, goal, est_data)

        xd = self._state_dimension

        # assume integrater uses first _state_dimension elements from est data
        # todo decide what to take based on type of goal
        state = np.vstack([
            est_data["cartesian_sensor_est"][comp][:xd]
                for comp in self.model.state_component
        ])
        
        # both state and goal is in [pos, vel, etc.]' with shape [T, ?, 1]
        state_goal = goal['goal']
        A = self.model.A(dt=dt)
        B = self.model.B(dt=dt)
        N = self.horizon

        # lifted system for tracking last state
        Abar = np.vstack([np.linalg.matrix_power(A, i) for i in range(1,N+1)])
        Bbar = np.vstack([
            np.hstack([
                np.hstack([np.linalg.matrix_power(A, p) @ B for p in range(row, -1, -1)]),
                np.zeros((xd, N-1-row))
            ]) for row in range(N)
        ])

        # tracking each state dim
        n_state_comp = len(self.model.state_component) # number of pos, vel, etc.
        traj = np.zeros((N, xd * n_state_comp, 1))
        for i in range(xd):
            # vector: pos, vel, etc. of a single dimension
            x = np.vstack([ state[ j * xd + i, 0 ] for j in range(n_state_comp) ])
            xref = np.vstack([ state_goal[ j * xd + i, 0 ] for j in range(n_state_comp) ])

            ubar = np.linalg.lstsq(
                a = Bbar[-xd:, :], b = xref - np.linalg.matrix_power(A, N) @ x)[0] # get solution

            xbar = (Abar @ x + Bbar @ ubar).reshape(N, n_state_comp, 1)

            for j in range(n_state_comp):
                traj[:, j * xd + i] = xbar[:, j]

        return traj


class CFSPlanner(IntegraterPlanner):

    def __init__(self, spec, model) -> None:
        super().__init__(spec, model)
    
    def _ineq(self, x, obs):
        '''
        inequality constraints. 
        constraints: ineq(x) > 0
        '''
        # norm distance restriction
        obs_p = obs.flatten()
        # obs_r = self.spec['obs_r']
        obs_r = 5 # todo tune
        obs_r = np.array(obs_r)
        
        # flatten the input x 
        x = x.flatten()
        dist = np.linalg.norm(x - obs_p) - obs_r
        return dist

    def _CFS(self, 
        x_ref,
        n_ob,
        obs_traj,
        cq = [10,0,10], 
        cs = [0,1,0.1], 
        minimal_dis = 0, 
        ts = 1, 
        maxIter = 30,
        stop_eps = 1e-3
    ):
        # without obstacle, then collision free
        
        if n_ob == 0 or len(obs_traj)==0: # no future obstacle information is provided 
            return np.array(x_ref)

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
        Q1 = np.identity(h * dimension)
        S1 = Q1
        # velocity term 
        Vdiff = np.identity(h*dimension) - np.diag(np.ones((1,(h-1)*dimension))[0],dimension)
        Q2 = np.matmul(Vdiff.transpose(),Vdiff) 
        # Acceleration term 
        Adiff = Vdiff - np.diag(np.ones((1,(h-1)*dimension))[0],dimension) + np.diag(np.ones((1,(h-2)*dimension))[0],dimension*2)
        Q3 = np.matmul(Adiff.transpose(),Adiff)
        # Vdiff = eye(nstep*dim)-diag(ones(1,(nstep-1)*dim),dim);

        # objective 
        Q = Q1*cq[0]+Q2*cq[1]+Q3*cq[2];
        S = S1*cs[0]+Q2*cs[1]+Q3*cs[2];

        # quadratic term
        H =  Q + S 
        # linear term
        f = -1 * np.dot(Q, x_origin)

        b = np.ones((h * n_ob, 1)) * (-minimal_dis)
        H = matrix(H,(len(H),len(H[0])),'d')
        f = matrix(f,(len(f), 1),'d')
        # b = matrix(b,(len(b),1),'d')

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

        # set the safety margin 
        D = 5

        # fig, ax = plt.subplots()
        # main CFS loop
        while dlt > stop_eps:
            cnt += 1
            Lstack, Sstack = [], []
            # inequality constraints 
            # l * x <= s
            Constraint = np.zeros((h * n_ob, len(x_rs)))
            
            for i in range(h):
                # get reference pos at time step i
                if i < h-1 and i > 0:
                    x_r = x_rs[i * dimension : (i + 1) * dimension] 

                    # get inequality value (distance)
                    # get obstacle at this time step 
                    obs_p = obs_traj[i,:]  
                    dist = self._ineq(x_r,obs_p)
                    # print(dist)

                    # get gradient 
                    ref_grad = jac_num(self._ineq, x_r, obs_p)
                    # print(ref_grad)

                    # compute
                    s = dist - D - np.dot(ref_grad, x_r)
                    l = -1 * ref_grad
                if i == h-1 or i == 0: # don't need inequality constraints for lst dimension 
                    s = np.zeros((1,1))
                    l = np.zeros((1,2))

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

            J = np.dot(np.transpose(x_ts - x_origin), np.dot(Q, (x_ts - x_origin))) + np.dot(np.transpose(x_ts), np.dot(S, x_ts))
            dlt = min(abs(J - J0), np.linalg.norm(x_ts - x_rs))
            J0 = J
            x_rs = x_ts
            if cnt >= maxIter:
                break
        
        # return the reference trajectory      
        x_rs = x_rs[: h * dimension]
        return x_rs.reshape(h, dimension)

    def _plan(self, dt: float, goal: dict, est_data: dict) -> np.array:
        
        xd = self.state_dimension
        N = self.horizon

        # get integrator interpolation
        traj = super()._plan(dt, goal, est_data).squeeze()

        # get obs relative pos
        obs_pos_list = []
        for name, info in est_data['obstacle_sensor_est'].items():
            if 'obs' in name:
                obs_pos_list.append(info['rel_pos'])
        
        state = est_data["cartesian_sensor_est"]['pos'][:xd]
        obs_traj = []
        for obs_pos in obs_pos_list:
            obs_traj.append(np.tile((state + obs_pos).reshape(1, -1), (N, 1))) # [N, xd]

        if len(obs_traj) > 1:
            obs_traj = np.concat(obs_traj, axis=-1) # [T, n_obs * xd]
        else:
            obs_traj = obs_traj[0]
        
        # CFS
        # mind = 0.0
        traj_pos_only = traj[:, :xd]
        traj_pos_safe = self._CFS(x_ref=traj_pos_only, n_ob=len(obs_pos_list), obs_traj=obs_traj)
        traj[:, :xd] = traj_pos_safe

        # for obs_pos in obs_pos_list:
        #     obs_pos_abs = obs_pos + state
        #     d = np.min(np.linalg.norm(
        #         traj[:, :self.state_dimension, 0] - obs_pos_abs.reshape(1, -1), axis=1)) - 5
        #     if d < mind:
        #         mind = d
        
        # if mind < 0.0:
        #     print("Collision: {}".format(mind))

        return traj[:, np.newaxis]


