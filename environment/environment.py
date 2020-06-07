import numpy as np

class BB8(object):
    def __init__(self, init_x, init_p):
        self.x = init_x
        self.p = init_p
    
    def f(self, x):
        return np.vstack([x[2], x[3], 0, 0])

    def g(self, x):
        return np.vstack([1, 1])

    def forward(self, u, dT):
        # dot_x = A * x + B * u
        dot_x = self.f(x) + self.g(x) * u
        self.p = self.p + (self.x[:2] + dot_x/2) * dT
        self.x = dot_x * dT
    

class World(object):
    def __init__(self, spec):
        self.spec = spec
        self.reset()
        self.agents = []
    
    def add_agent(self, agent, agent_env_spec):
        self.agents.append(BB8(agent_env_spec['init_x'], agent_env_spec['init_p']))

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
            self.world.add_agent(agents[i], env_spec['agent_env_spec'][i])
            for j in range(len(agents[i].sensors)):
                self.world.add_sensor(agents[i].sensors[j].spec)

    def reset(self):
        env_info, sensor_data = self.world.reset()
        return env_info, sensor_data
    def step(self, control):
        env_info, sensor_data = self.world.simulate(control)
        return env_info, sensor_data
