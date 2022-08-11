from abc import ABC, abstractmethod
from operator import concat
from matplotlib.pyplot import axis
import numpy as np
np.set_printoptions(suppress=True)
from cvxopt import matrix, solvers
from matplotlib import pyplot as plt

solvers.options['show_progress'] = False
c_white = np.array([1, 1, 1])
c_red = np.array([1, 0, 0])
c_green = np.array([0, 1, 0])
c_scale_min = 0.2
c_scale_max = 1.0

def BlackBoxPrediction(traj):

    obs_traj = []

    for pos in traj:
        ax = pos[0]
        ay = pos[1]
        ak = np.exp(np.sqrt((ax-40)**2+(ay-20)**2) / 8)
        ak = np.exp(ak)
        ak = np.min([ak, 10])
        obs_traj.append([
            ax + ak * np.cos(ax*0.1+np.pi/6),
            ay + ak * np.sin(ax*0.1+np.pi/6)
        ])
        
    obs_traj = np.array(obs_traj)

    return obs_traj

def vstack_wrapper(a, b):
    if a == []:
        return b
    else:
        x = np.vstack((a,b))
        return x

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

        self.blackbox_prediction = spec['blackbox_prediction']
        self.viz_CFS_iter = spec['viz_CFS_iter']
        self.viz_prediction_planning = spec['viz_prediction_planning']
        self.max_prediction_planning_iter = spec['max_prediction_planning_iter']
    
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
        ts = 1, 
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
        # b = matrix(b,(len(b),1),'d')

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
        D = 5

        # setup plot
        if self.viz_CFS_iter:
            # plot obstacle
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.axis('equal')
            ax.set(xlim=(0, 101), ylim=(0, 101))
            for i in range(h):
                for obs_i in range(n_obs):
                    circ = plt.Circle(
                        obs_traj[i, obs_i * dimension : (obs_i+1) * dimension],
                        5.0, color='k', clip_on=False, fill=False)
                    ax.add_patch(circ)
            # plot start end
            ax.plot(x_ref[:, 0], x_ref[:, 1], color=tuple(c_green*c_scale_min+c_red*(1-c_scale_min)))
            # plot buffer
            plot_buf = []

        # main CFS loop
        while dlt > stop_eps:

            cnt += 1

            # inequality constraints 
            # A * x <= b
            A, b = [], []
            
            # TODO: construct A, b matrices
            return None

            # --------------------------------- solution --------------------------------- #
            
            # TODO

            # ------------------------------- solution ends ------------------------------ #

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

            # add to plot
            if self.viz_CFS_iter:
                plot_buf.append(x_sol.reshape(h, 2))

            if cnt >= maxIter:
                break
        
        # draw
        if self.viz_CFS_iter:
            c_step = (c_scale_max - c_scale_min) / len(plot_buf)
            for iter_i, traj in enumerate(plot_buf):
                c_scale = c_scale_min + c_step * (iter_i + 1)
                ax.plot(traj[:, 0], traj[:, 1], color=tuple(c_green*c_scale+c_red*(1-c_scale)))
            fig.canvas.draw()
            plt.pause(0.001)
            print(f'Drew {len(plot_buf)} routes.')

        # return the reference trajectory      
        x_sol = x_sol[: h * dimension]

        return x_sol.reshape(h, dimension)

    def _planning_prediction_converge(self, obs_traj, traj):
        for obs_pt, pt in zip(obs_traj, traj):
            if self._ineq(pt, obs_pt) < 0:
                print(pt)
                print(obs_pt)
                return False
        return True

    def _plan(self, dt: float, goal: dict, est_data: dict) -> np.array:
        
        xd = self.state_dimension
        N = self.horizon

        # get integrator interpolation as nominal path
        traj = super()._plan(dt, goal, est_data).squeeze()

        # get obs relative pos
        obs_pos_list = []
        for name, info in est_data['obstacle_sensor_est'].items():
            if 'obs' in name:
                obs_pos_list.append(info['rel_pos'])
        
        # ------------------------- planning with prediction ------------------------- #
        traj_pos_only = traj[:, :xd]
        prediction_planning_cnt = 0
        while prediction_planning_cnt < self.max_prediction_planning_iter:
            # Get obstacle trajectory (either blackbox or static)
            if self.blackbox_prediction:
                # use blackbox function
                assert(len(obs_pos_list) == 1) # for now only single object with blackbox pred

                # TODO use blackbox prediction
                obs_traj = []
                
                # --------------------------------- solution --------------------------------- #

                # TODO

                # ------------------------------- solution ends ------------------------------ #

                prediction_planning_cnt += 1

                # setup plot
                if self.viz_prediction_planning:
                    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                    ax.set_title(f'Planning with prediction iteration {prediction_planning_cnt}')
                    ax.axis('equal')
                    ax.set(xlim=(0, 101), ylim=(0, 101))
                    for i in range(len(obs_traj)):
                        circ = plt.Circle(
                            obs_traj[i], 5.0, color='k', clip_on=False, fill=False)
                        ax.add_patch(circ)
                    # plot start end
                    ax.plot(traj_pos_only[:, 0], traj_pos_only[:, 1], color=tuple(c_red))

            else:
                obs_traj = []
                # get obs absolute pos
                state = est_data["cartesian_sensor_est"]['pos'][:xd]
                for obs_pos in obs_pos_list:
                    obs_traj.append(np.tile((state + obs_pos).reshape(1, -1), (N, 1))) # [N, xd]

                if len(obs_traj) > 1:
                    obs_traj = np.concatenate(obs_traj, axis=-1) # [N, xd * n_obs]
                elif len(obs_traj) == 1:
                    obs_traj = obs_traj[0]
            
            # CFS
            traj_pos_only = self._CFS(x_ref=traj_pos_only, n_obs=len(obs_pos_list), obs_traj=obs_traj)

            if traj_pos_only is None or not self.blackbox_prediction:
                break

            # plot
            if self.blackbox_prediction and self.viz_prediction_planning:
                ax.plot(traj_pos_only[:, 0], traj_pos_only[:, 1], color=tuple(c_green))
                fig.canvas.draw()
                plt.pause(3.0)
            
            # planning with prediction convergence condition
            if self._planning_prediction_converge(BlackBoxPrediction(traj_pos_only), traj_pos_only):
                break

        print(f"Plan pred {prediction_planning_cnt} iters.")

        # update traj
        if traj_pos_only is not None:
            traj[:, :xd] = traj_pos_only
            # approximate velocity
            traj[:-1, -xd:] = (traj_pos_only[1:, :] - traj_pos_only[:-1, :]) / dt
            traj[-1, -xd:] = 0.0

        return traj[:, np.newaxis]


