from abc import ABC, abstractmethod
import numpy as np

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
    
    def _plan(self, dt: float, goal: dict, est_data: dict) -> np.array:
        
        xd = self.state_dimension

        # get integrator interpolation
        traj = super()._plan(dt, goal, est_data)

        # invoke CFS to avoid collision
        # ! tmp logic for sanity check

        # get obs relative pos
        obs_pos_list = []
        for name, info in est_data['obstacle_sensor_est'].items():
            if 'obs' in name:
                obs_pos_list.append(info['rel_pos'])
        
        # check collision along traj, to be replaced by CFS call
        mind = 0.0
        state = est_data["cartesian_sensor_est"]['pos'][:xd]
        for obs_pos in obs_pos_list:
            obs_pos_abs = obs_pos + state
            d = np.min(np.linalg.norm(
                traj[:, :self.state_dimension, 0] - obs_pos_abs.reshape(1, -1), axis=1)) - 5
            if d < mind:
                mind = d
        
        if mind < 0.0:
            print("Collision: {}".format(mind))

        return traj


