import os
import pybullet as p
import pybullet_data

import numpy as np
import matplotlib.pyplot as plt
import importlib
import math

class BulletEnv():
    metadata = {'render.modes': ['human']}

    def __init__(self, env_spec, comp_agents):
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        self.dt = env_spec['dt']
        WorldClass = getattr(importlib.import_module("env.bullet_world"), env_spec["world"]["type"])
        self.world = WorldClass(env_spec["world"]["spec"])
        self.env_spec = env_spec
        self.comp_agents = comp_agents
        self.reset()

    def step(self, actions):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        self.world.simulate(actions, self.dt)
        env_info, sensor_data = self.world.measure()
        self.render(env_info)
        return self.dt, env_info, sensor_data

    def reset(self):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything
        self.world.reset()
        for i in range(len(self.comp_agents)):
            self.world.add_agent(self.comp_agents[i], self.env_spec['agent_env_spec'][self.comp_agents[i].name])
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1) # rendering's back on again
        env_info, sensor_data = self.world.measure()
        return self.dt, env_info, sensor_data

    def render(self, env_info, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
                                                          distance=1.,
                                                          yaw=90,
                                                          pitch=-70,
                                                          roll=0,
                                                          upAxisIndex=2)

        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                   aspect=float(960) /720,
                                                   nearVal=0.1,
                                                   farVal=100.0)
        
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                            height=720,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        p.disconnect()
