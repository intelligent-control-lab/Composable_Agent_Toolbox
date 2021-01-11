from env.carla_world import CarlaWorld


class CarlaEnv:
    """Interface to the Carla World"""
    def __init__(self, env_spec, comp_agents):
        self.dt = env_spec['dt']
        self.env_spec = env_spec
        self.comp_agents = comp_agents
        self.world = CarlaWorld(env_spec["world"]["spec"])

    def reset(self):
        self.world.reset()
        env_info, sensor_data = self.world.measure()
        return self.dt, env_info, sensor_data

    def step(self, actions, render=True):
        self.world.simulate(actions, self.dt)
        env_info, sensor_data = self.world.measure()
        if render:
            self.render(env_info)
        return self.dt, env_info, sensor_data

    def render(self, env_info):
        self.world.render()
