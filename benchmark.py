
class World(object):
    def __init__(self, spec):
        self.spec = spec
        self.reset()
    
    def add_agent(self, agent):
        pass

    def add_sensor(self, agents):
        pass

    def reset(self):
        self.cache = None
        env_info = {1:1, 2:2}
        sensor_data = [1,2]
        agent_sensor_data = [sensor_data, sensor_data]
        return env_info, agent_sensor_data
    
    def simulate(self, control):
        env_info = {1:1, 2:2}
        sensor_data = [1,2]
        agent_sensor_data = [sensor_data, sensor_data]
        return env_info, agent_sensor_data

class Environment(object):
    def __init__(self, env_spec, agents):
        '''
        Each environment has several pre-defined robot classes and sensor
        classes. The add_agent function will instantiate a robot class and 
        some sensors based on the specs.
        '''
        self.world = World(env_spec)
        for i in range(len(agents)):
            self.world.add_agent(agents[i])
            for j in range(len(agents[i].sensors)):
                self.world.add_sensor(agents[i].sensors[j].spec)

    def reset(self):
        env_info, sensor_data = self.world.reset()
        return env_info, sensor_data
    def step(self, control):
        env_info, sensor_data = self.world.simulate(control)
        return env_info, sensor_data



class Agent(object):
    def __init__(self, module_spec):
       self.instantiate_by_spec(module_spec)
    
    def instantiate_by_spec(self, module_spec):
        self.task = module_spec['task']
        self.model = Model(module_spec['model'])
        self.estimator = Estimator(module_spec['estimator'], self.model)
        self.planner = Planner(module_spec['planner'], self.model)
        self.controller = Controller(module_spec['controller'], self.model)
        self.sensors = []
        for i in range(len(module_spec['sensors'])):
            self.sensors.append(Sensor(module_spec['sensors'][i]))

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

class Sensor(object):
    def __init__(self, spec):
        self.spec = spec

#Jaskaran
class Estimator(object):
    def __init__(self, spec, model):
        self.spec = spec
        self.model = model
        self.pred_state = None
        self.cache = {}

    def fusion(self, sensor_data):
        fus_sensor_data = sensor_data
        return fus_sensor_data

    def estimate(self, sensor_data):
        fus_sensor_data = self.fusion(sensor_data)
        dT = sensor_data[0]
        est_param = 1
        self.model.param = est_param
        est_state = fus_sensor_data
        return est_state, est_param
    
#Weiye
class Planner(object):
    def __init__(self, spec, model):
        self.spec = spec
        self.model = model
        self.cache = {}
    def planning(self, dt, goal, agent_state):
        agent_next_state = agent_state
        return agent_next_state
    def re_planning(self, dt, goal, agent_state):
        agent_next_state = agent_state
        return agent_next_state

#Suqin
class Controller(object):
    def __init__(self, spec, model):
        self.spec = spec
        self.model = model

    def control(self, dt, x, goal_x):
        traj_c = self.traj_control(dt, x, goal_x)
        safe_c = self.safe_control(dt, x, goal_x)
        return self.merge(traj_c, safe_c)
    
    def traj_control(self, dt, x, goal_x):
        return 1
    
    def safe_control(self, dt, x, goal_x):
        return 1

    def merge(self, traj_c, safe_c):
        return 2

#Chase
class Model(object):
    def __init__(self, spec):
        pass
    def forward(self, x):
        '''
        dot_x = f(x) + g(x) u
        '''
        fx = 1
        gx = 1
        return fx, gx

    def inverse(self, p, x):
        '''
        dp_dx: derivative of the robot's cartesian state to its internal state,
               in the dodge obstacle task, the cartesian state is set as the
               closed point on the robot to the obstacle.
        '''
        dp_dx = 1
        return dp_dx
        

class Evaluator(object):
    def __init__(self, agent_specs, env_spec):
        self.agent_specs = agent_specs
        self.env_spec = env_spec

    def evaluate(self, record):
        return 0

if __name__ == "__main__":

    module_spec = {
        "task": {(1,1):10, (2,2):20},
        "model": "Model",
        "estimator": "Estimator",
        "planner": "Planner",
        "controller": "Controller",
        "sensors": ["Sensor", "Sensor"],
    }
    agent_specs = [module_spec, module_spec]
    env_spec = {
        "friction": 0
    }
    evaluator = Evaluator(agent_specs, env_spec)

    agents = []
    for i in range(len(evaluator.agent_specs)):
        agents.append(Agent(evaluator.agent_specs[i]))

    env = Environment(env_spec, agents)
    env_info, agent_sensor_data = env.reset()
    record = []

    
    for it in range(100):
        print("iter = ",it)

        actions = []
        for i in range(len(agents)):
            actions.append(agents[i].action(agent_sensor_data[i]))
            #sensor data is grouped by agent
        env_info, agent_sensor_data = env.step(actions)
        record.append(env_info)

    evaluator.evaluate(record)

