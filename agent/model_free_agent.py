from agent import sensor
import numpy as np
import importlib
from .agent_base import AgentBase
import torch
from .rl_modules.models import actor

#Notice: The model we use here is not trained on our environment. It is just for illustration use.
class ModelFreeAgent(AgentBase):
    def __init__(self, spec):

        self.name = spec["name"]
        
        # load the model param
        model_path = spec["model_path"]
        o_mean, o_std, g_mean, g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.actor_network = actor(spec["env_params"])
        self.actor_network.load_state_dict(model)
        self.actor_network.eval()

        self.sensors    = {}
        
        for i in range(len(spec["sensors"])):
            self.sensors[spec["sensors"][i]["spec"]["alias"]] = sensor.Sensor(spec["sensors"][i])

    def action(self, dt, sensors_data):
        inputs = np.vstack([sensors_data["state_sensor"]["state"], sensors_data["cartesian_sensor"]["pos"], sensors_data["goal_sensor"]["rel_pos"]])
        inputs = torch.tensor(inputs.T, dtype=torch.float32)

        with torch.no_grad():
            pi = self.actor_network(inputs)
        
        control = pi.detach().numpy().squeeze()
        control = control[:3]

        ret = {"control"  : control}
        
        if "communication_sensor" in self.sensors.keys():
            ret["broadcast"] = {
                "state:":sensors_data["state_sensor"]["state"]
            }

        return ret

