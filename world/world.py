import numpy as np
import world.sensor

class BB8(object):
    def __init__(self, init_x, init_p):
        self._x = init_x
        self._p = init_p
    
    def _f(self, x):
        return np.vstack([x[2], x[3], 0, 0])

    def _g(self, dt):
        B = np.matrix(np.zeros((4,2)))
        B[0,0] = dt/2
        B[1,1] = dt/2
        B[2,0] = 1
        B[3,1] = 1
        return B

    def forward(self, u, dt):
        # x = [x y dx dy], u = [ax ay]
        dot_x = self._f(self._x) + self._g(dt) * np.vstack(u)
        self._x = self._x + dot_x * dt
        self._p = self._x[[0,1]]
    
    @property
    def pos(self):
        return self._x[[0,1]]
    
    @property
    def vel(self):
        return self._x[[2,3]]

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
    
    def simulate(self, controls, dt):
        env_info = {1:1, 2:2}
        sensor_data = [1,2]
        agent_sensor_data = [sensor_data, sensor_data]
        for i,agent in enumerate(self.agents):
            agent.forward(controls[i], dt)
            print(agent.pos)
            print(agent.vel)
        return env_info, agent_sensor_data
