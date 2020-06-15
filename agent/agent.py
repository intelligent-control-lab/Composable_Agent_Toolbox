import sensor, estimator, planner, controller, model
import numpy as np
class Agent(object):
    def __init__(self, module_spec):
       self.instantiate_by_spec(module_spec)
       self.replanning_cycle = 10
       self.replanning_timer = self.replanning_cycle
    
    def instantiate_by_spec(self, module_spec):
        self.task = module_spec['task']
        self.model = model.Model(module_spec['model'])
        self.estimator = estimator.Estimator(module_spec['estimator'], self.model)
        self.planner = planner.Planner(module_spec['planner'], self.model)
        self.controller = controller.Controller(module_spec['controller'], self.model)
        self.sensors = []
        for i in range(len(module_spec['sensors'])):
            self.sensors.append(sensor.Sensor(module_spec['sensors'][i]))

    def get_goal(self, task_state):
        return np.zeros((4,1))

    def action(self, dt, sensors_data):
        est_state, est_parameter = self.estimator.estimate(sensors_data[0])
        goal = self.get_goal((est_state, est_state))

        if self.replanning_timer == self.replanning_cycle:
            self.planned_traj = self.planner.planning(dt, goal, est_state)
            self.replanning_timer = 0
        agent_goal_state = self.planned_traj[self.replanning_timer]
        self.replanning_timer += 1

        control = self.controller.control(dt, est_state, agent_goal_state, est_parameter)

        return control