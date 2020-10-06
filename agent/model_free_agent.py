from agent import sensor
import numpy as np
import importlib
from .agent_base import AgentBase

#Notice: The model we use here is not trained on our environment. It is just for illustration use.
class ModelFreeAgent(AgentBase):
    def __init__(self, spec):

        self.name = spec["name"]
        self.sensors    = {}

        for i in range(len(spec["sensors"])):
            self.sensors[spec["sensors"][i]["spec"]["alias"]] = sensor.Sensor(spec["sensors"][i])

        PolicyClass = getattr(importlib.import_module("agent.policy"), spec["policy"]["type"])
        self.policy = PolicyClass(spec["policy"]["spec"])
        
    def action(self, dt, sensors_data):
        control = self.policy.action(dt, sensors_data)

        ret = {"control"  : control}
        
        if "communication_sensor" in self.sensors.keys():
            ret["broadcast"] = {
                "state:":sensors_data["state_sensor"]["state"]
            }

        return ret
