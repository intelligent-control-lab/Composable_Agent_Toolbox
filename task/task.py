import numpy as np

class Task(object):
    def __init__(self, spec, model):
        pass

    def goal(self, x):
        return np.zeros((4,1))

class ReachingTask(Task):
    def __init__(self, spec, model):
        super().__init__(spec, model)
        self.eps = spec["reaching_eps"]
        goal_range = spec["goal_range"]
        self.goal_starts = np.vstack(goal_range[0])
        self.goal_widths = np.vstack(goal_range[1]) - np.vstack(goal_range[0])
        self.goal_shape = np.shape(self.goal_starts)
        self._generate_goal()
        
    def _generate_goal(self):
        """Generate a goal([x,y,vel_x,vel_y]) for the agent.
        """
        self.current_goal = np.random.rand(*(self.goal_shape)) * self.goal_widths + self.goal_starts

    def goal(self, est_data):
        """Return the current goal for the agent.
        """
        pos_vel = np.vstack([est_data["cartesian_sensor_est"]["pos"], est_data["cartesian_sensor_est"]["vel"]])
        if np.max(np.abs(pos_vel - self.current_goal)) < self.eps:
            self._generate_goal()
        return self.current_goal
