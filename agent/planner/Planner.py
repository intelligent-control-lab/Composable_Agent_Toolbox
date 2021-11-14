from abc import ABC, abstractmethod
import numpy as np

class Planner(ABC):
    def __init__(self, spec, model) -> None:
        self.spec = spec
        self.model = model
        self.replanning_cycle = spec["replanning_cycle"]
        self.horizon = spec["horizon"]
        self.xdim = spec["state_dimension"]
    
    @property
    def state_dimension(self):
        return self.xdim

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

    def __init__(self, spec, model) -> None:
        super().__init__(spec, model)
    
    def _plan(self, dt: float, goal: dict, est_data: dict) -> np.array:
        super()._plan(dt, goal, est_data)

        # assume data from est has correct state_dimension and info for integrator model
        # ! for now by default cartesian
        state = np.vstack(
            [est_data["cartesian_sensor_est"][comp] for comp in self.model.state_component])
        
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
                np.zeros((self.xdim, N-1-row))
            ]) for row in range(N)
        ])

        # tracking each state dim
        n_state_comp = len(self.model.state_component) # number of pos, vel, etc.
        traj = np.zeros((N, self.xdim * n_state_comp, 1))
        for i in range(self.xdim):
            # vector: pos, vel, etc. of a single dimension
            x = np.vstack([ state[ j * self.xdim + i, 0 ] for j in range(n_state_comp) ])
            xref = np.vstack([ state_goal[ j * self.xdim + i, 0 ] for j in range(n_state_comp) ])

            ubar = np.linalg.lstsq(
                a = Bbar[-self.xdim:, :], b = xref - np.linalg.matrix_power(A, N) @ x)[0] # get solution

            xbar = (Abar @ x + Bbar @ ubar).reshape(N, n_state_comp, 1)

            for j in range(n_state_comp):
                traj[:, j * self.xdim + i] = xbar[:, j]
        
        return traj

# todo adapt CFS planner as a safe planner, also use planning model
# state planner by default assume no structure about model (only linearize)
