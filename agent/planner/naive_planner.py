import numpy as np
class NaivePlanner(object):
    def __init__(self, spec, model):
        self.spec = spec
        self.model = model
        self.cache = {}
        self.replanning_cycle = spec["replanning_cycle"]
        self.horizon = spec["horizon"]

    def planning(self, dt, goal, est_data):
        pos_vel = np.vstack([est_data["cartesian_sensor_est"]["pos"], est_data["cartesian_sensor_est"]["vel"]])
        traj = []
        goal = goal['goal']
        frac = (goal - pos_vel)*1./self.horizon
        for i in range(self.horizon):
            traj.append(pos_vel + frac*i)
        return np.array(traj)

    # todo remove arm
    def planning_arm(self, dt, goal, est_data):
        pos_vel = np.vstack([est_data["cartesian_sensor_est"]["pos"], est_data["cartesian_sensor_est"]["vel"]])
        traj = []
        goal = goal['goal']
        frac = (goal - pos_vel)*1./self.horizon
        for i in range(self.horizon):
            traj.append(pos_vel + frac*i)
        return np.array(traj)

# todo double integrator

# todo adapt naive planner to use planning model

# ! return AB matrices from input state x

# todo adapt CFS planner as a safe planner, also use planning model

# todo cartesian planner by default uses integrator
# todo state planner by default assume no structure about model (only linearize)
