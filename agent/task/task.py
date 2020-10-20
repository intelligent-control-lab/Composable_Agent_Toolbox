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

class FrankaReachingDoubleGoalTask(Task):
    def __init__(self, spec, model):
        self.state_goal_list = spec["state_goal_list"]
        self.cartesian_goal_list = spec["cartesian_goal_list"]
        print(self.cartesian_goal_list)
        print(self.state_goal_list)
    def goal(self, est_data):
        """Return the goal positon as a reference for the planner.
        """
        # return est_data["goal_sensor_est"]["rel_pos"] + est_data["cartesian_sensor_est"]["pos"]
        goal_rel_pos_vel = np.vstack([est_data["goal_sensor_est"]["rel_pos"], est_data["goal_sensor_est"]["rel_vel"]])
        self_abs_pos_vel = np.vstack([est_data["cartesian_sensor_est"]["pos"], est_data["cartesian_sensor_est"]["vel"]])
        goal_pos = goal_rel_pos_vel + self_abs_pos_vel
        idx = 0
        
        goal_pos = goal_pos[:3].reshape((1,3))

        for i in range(len(self.cartesian_goal_list)):
            # print('hiua')
            # print(goal_pos - self.cartesian_goal_list[i])
            if np.linalg.norm(goal_pos - self.cartesian_goal_list[i]) < np.linalg.norm(goal_pos - self.cartesian_goal_list[idx]):
                idx = i
        return self.state_goal_list[idx]
