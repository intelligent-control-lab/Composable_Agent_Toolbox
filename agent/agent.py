import sensor, estimator, planner, controller, model
class Agent(object):
    def __init__(self, module_spec):
       self.instantiate_by_spec(module_spec)
    
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
        goal = self.task[task_state]
        return goal

    def action(self, sensor_data):
        est_obs, est_parameter = self.estimator.estimate(sensor_data)
        dt, est_state = est_obs
        goal = self.get_goal((est_state, est_obs[1]))
        agent_goal_state = self.planner.planning(dt, goal, est_state)
        control = self.controller.control(dt, est_state, agent_goal_state)

        return control