import importlib
from agent.model_based_agent import ModelBasedAgent
from agent import sensor


import numpy as np


class MASAgent():

    def __init__(self, module_spec: dict) -> None:
        self._instantiate_by_spec(module_spec)
        self.path = []

    def _class_by_name(self, module_name, class_name):
        """Return the class handle by name of the class
        """
        module_name = "agent." + module_name
        ModuleClass = getattr(importlib.import_module(module_name), class_name)
        return ModuleClass

    def _instantiate_by_spec(self, module_spec):
        """Instantiate modules based on user given specs
        """
        self.module_spec    = module_spec # for access by mp_wrapper
        self.name           = module_spec["name"]
        self.planning_model = self._class_by_name("model",      module_spec["model"]["planning"]["type"])(module_spec["model"]["planning"]["spec"])
        self.control_model  = self._class_by_name("model",      module_spec["model"]["control" ]["type"])(module_spec["model"]["control" ]["spec"])
        self.task           = self._class_by_name("task",       module_spec["task"      ]["type"])(module_spec["task"       ]["spec"], self.planning_model)
        self.estimator      = self._class_by_name("estimator",  module_spec["estimator" ]["type"])(module_spec["estimator"  ]["spec"], self.planning_model)
        self.planner        = self._class_by_name("planner",    module_spec["planner"   ]["type"])(module_spec["planner"    ]["spec"], self.planning_model)
        self.controller     = self._class_by_name("controller", module_spec["controller"]["type"])(module_spec["controller" ]["spec"], self.control_model)
        self.sensors        = {}
        
        for i in range(len(module_spec["sensors"])):
            self.sensors[module_spec["sensors"][i]["spec"]["alias"]] = sensor.Sensor(module_spec["sensors"][i])

    def next_point(self) -> np.array:
        pass

    def update_path(self, path: list[np.array]) -> None:
        self.path = path

    def control(self) -> None:
        pass


