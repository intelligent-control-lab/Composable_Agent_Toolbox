from abc import ABC, abstractmethod
from operator import concat
from matplotlib.pyplot import axis
import numpy as np
np.set_printoptions(suppress=True)
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False

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

    def _plan(self, dt: float, goal: dict, est_data: dict) -> np.array:

        super()._plan(dt, goal, est_data)

        pos_vel = np.vstack([est_data["cartesian_sensor_est"]["pos"], est_data["cartesian_sensor_est"]["vel"]])
        traj = []
        goal_pos_vel = goal['goal']
        frac = (goal_pos_vel - pos_vel)*1./self.horizon
        for i in range(self.horizon):
            traj.append(pos_vel + frac*i)

        return np.array(traj)

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
        obs_r = 5
        obs_r = np.array(obs_r)
        
        # flatten the input x 
        x = x.flatten()
        dist = np.linalg.norm(x - obs_p) - obs_r

        return dist

    def _CFS(self, 
        x_ref,
        n_obs,
        obs_traj,
        cq = [10,0,10], 
        cs = [0,1,0.1], 
        minimal_dis = 0, 
        maxIter = 30,
        stop_eps = 1e-3
    ):
        # without obstacle, then collision free

        if n_obs == 0 or len(obs_traj)==0: # no future obstacle information is provided 
            return np.array(x_ref)

        # has obstacle, the normal CFS procedure 

        # planning parameters 
        h = x_ref.shape[0] # horizon
        dimension = x_ref.shape[1] # state dimension

        # flatten the trajectory to one dimension
        # flatten to one dimension for applying qp, in the form of x0,y0,x1,y1,...
        x_ref_vec = x_ref.reshape(-1, 1)

        # decision variable
        x_sol = x_ref_vec.copy()

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
        f = -1 * np.dot(Q, x_ref_vec)

        b = np.ones((h * n_obs, 1)) * (-minimal_dis)
        H = matrix(H,(len(H),len(H[0])),'d')
        f = matrix(f,(len(f), 1),'d')

        # reference trajctory cost
        J0 =  (x_sol - x_ref_vec).T @ Q @ (x_sol - x_ref_vec) + x_sol.T @ S @ x_sol # initial cost
        J = float('inf') # new cost
        dlt = float('inf') # improvement
        cnt = 0 # iteration count

        # equality constraints 
        # start pos and end pos remain unchanged 
        Aeq = np.zeros((dimension*2, len(x_ref_vec)))
        for i in range(dimension):
            Aeq[i,i] = 1
            Aeq[dimension*2-i-1, len(x_ref_vec)-i-1] = 1
        
        beq = np.zeros((dimension*2, 1))
        beq[0:dimension,0] = x_ref_vec[0:dimension,0]
        beq[dimension:dimension*2, 0] = x_ref_vec[dimension*(h-1): dimension*h, 0]

        # transform to convex optimization matrix 
        Aeq = matrix(Aeq,(len(Aeq),len(Aeq[0])),'d')
        beq = matrix(beq,(len(beq),1),'d')

        # set the safety margin 
        D = 3

        # main CFS loop
        while dlt > stop_eps:

            cnt += 1

            # inequality constraints 
            # A * x <= b
            A, b = [], []
            
            # ------------------------------ write your code ----------------------------- #
            # TODO: construct A, b matrices that represent the CFS
            # TODO: remove 'return None' to test your code
            return None
            # ------------------------------------- - ------------------------------------ #

            # convert to matrix
            A = matrix(A,(len(A),len(A[0])),'d')
            b = matrix(b,(len(b),1),'d')

            # QP solver 
            sol = solvers.qp(H, f, A, b, Aeq, beq)
            x_sol_new = sol['x']
            x_sol_new = np.reshape(x_sol_new, (len(x_sol),1))
            
            # check convergence
            J = (x_sol_new - x_ref_vec).T @ Q @ (x_sol_new - x_ref_vec) + x_sol_new.T @ S @ x_sol_new
            dlt = min(abs(J - J0), np.linalg.norm(x_sol_new - x_sol))
            J0 = J
            x_sol = x_sol_new
            if cnt >= maxIter:
                break
        
        # return the reference trajectory      
        x_sol = x_sol[: h * dimension]

        return x_sol.reshape(h, dimension)

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
        
        # get obs absolute pos
        state = est_data["cartesian_sensor_est"]['pos'][:xd]
        obs_traj = []
        for obs_pos in obs_pos_list:
            obs_traj.append(np.tile((state + obs_pos).reshape(1, -1), (N, 1))) # [N, xd]

        if len(obs_traj) > 1:
            obs_traj = np.concatenate(obs_traj, axis=-1) # [N, xd * n_obs]
        elif len(obs_traj) == 1:
            obs_traj = obs_traj[0]
        
        # CFS
        traj_pos_only = traj[:, :xd]
        traj_pos_safe = self._CFS(x_ref=traj_pos_only, n_obs=len(obs_pos_list), obs_traj=obs_traj)
        
        if traj_pos_safe is not None:
            traj[:, :xd] = traj_pos_safe
            # approximate velocity
            traj[:-1, -xd:] = (traj_pos_safe[1:, :] - traj_pos_safe[:-1, :]) / dt
            traj[-1, -xd:] = 0.0

        return traj[:, np.newaxis]


