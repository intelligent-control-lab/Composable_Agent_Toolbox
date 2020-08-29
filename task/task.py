import numpy as np

class Task(object):
    def __init__(self, spec, model):
        pass

    def goal(self, x):
        return np.zeros((4,1))


class FlatReachingTask(Task):

    def goal(self, est_data):
        """Return the goal positon as a reference for the planner.
        """
        goal_rel_pos_vel = np.vstack([est_data["goal_sensor_est"]["rel_pos"], est_data["goal_sensor_est"]["rel_vel"]])
        self_abs_pos_vel = np.vstack([est_data["cartesian_sensor_est"]["pos"], est_data["cartesian_sensor_est"]["vel"]])
        return goal_rel_pos_vel + self_abs_pos_vel


class FrankaReachingTask(Task):

    def goal(self, est_data):
        """Return the goal positon as a reference for the planner.
        """
        
        # return est_data["goal_sensor_est"]["rel_pos"] + est_data["cartesian_sensor_est"]["pos"]
        goal_rel_pos_vel = np.vstack([est_data["goal_sensor_est"]["rel_pos"], est_data["goal_sensor_est"]["rel_vel"]])
        self_abs_pos_vel = np.vstack([est_data["cartesian_sensor_est"]["pos"], est_data["cartesian_sensor_est"]["vel"]])
        return goal_rel_pos_vel + self_abs_pos_vel
