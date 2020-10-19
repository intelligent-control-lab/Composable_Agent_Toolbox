from agent import sensor
import numpy as np
import importlib
from .agent_base import AgentBase

class ModelBasedAgent(AgentBase):
    def __init__(self, module_spec):
        self.instantiate_by_spec(module_spec)
        self.replanning_timer = self.planner.replanning_cycle
        self.last_control     = [0.0,0.0]
        
    def _class_by_name(self, module_name, class_name):
        """Return the class handle by name of the class
        """
        module_name = "agent." + module_name
        ModuleClass = getattr(importlib.import_module(module_name), class_name)
        return ModuleClass

    def instantiate_by_spec(self, module_spec):
        """Instantiate modules based on user given specs
        """
        self.name = module_spec["name"]
        self.model      = self._class_by_name("model",      module_spec["model"     ]["type"])(module_spec["model"      ]["spec"])
        self.task       = self._class_by_name("task",       module_spec["task"      ]["type"])(module_spec["task"       ]["spec"], self.model)
        self.estimator  = self._class_by_name("estimator",  module_spec["estimator" ]["type"])(module_spec["estimator"  ]["spec"], self.model)
        self.planner    = self._class_by_name("planner",    module_spec["planner"   ]["type"])(module_spec["planner"    ]["spec"], self.model)
        self.controller = self._class_by_name("controller", module_spec["controller"]["type"])(module_spec["controller" ]["spec"], self.model)
        self.sensors    = {}
        
        for i in range(len(module_spec["sensors"])):
            self.sensors[module_spec["sensors"][i]["spec"]["alias"]] = sensor.Sensor(module_spec["sensors"][i])

    def action(self, dt, sensors_data):
        u = self.last_control
        est_data, est_param = self.estimator.estimate(u,sensors_data)
        goal = self.task.goal(est_data)
        if self.replanning_timer == self.planner.replanning_cycle:
            # add the future planning information for another agent 
            self.planned_traj = self.planner.planning_arm(dt, goal, est_data)
            # self.planned_traj = self.planner.planning(dt, goal, est_data)
            
            self.replanning_timer = 0
        next_traj_point = self.planned_traj[min(self.replanning_timer, self.planned_traj.shape[0]-1)]  # After the traj ran out, always use the last traj point for reference.
        next_traj_point = np.expand_dims(next_traj_point, axis=0).T
        self.replanning_timer += 1

        control = self.controller.control(dt, est_data, next_traj_point, est_param)
        
        self.last_control = control
        ret = {"control"  : control}
        
        if "communication_sensor" in self.sensors.keys():
            ret["broadcast"] = {
                "planned_traj":self.planned_traj[min(self.replanning_timer, self.planner.horizon-1):],
                "state":est_param["ego_state_est"]
            }

        return ret



class UserControlAgent(AgentBase):
    def __init__(self, module_spec):
        self.instantiate_by_spec(module_spec)
        self.last_control     = [0.0,0.0]
        
    def _class_by_name(self, module_name, class_name):
        """Return the class handle by name of the class
        """
        ModuleClass = getattr(importlib.import_module(module_name), class_name)
        return ModuleClass

    def instantiate_by_spec(self, module_spec):
        """Instantiate modules based on user given specs
        """
        self.name = module_spec["name"]
        self.model      = self._class_by_name("model",      module_spec["model"     ]["type"])(module_spec["model"      ]["spec"])
        self.estimator  = self._class_by_name("estimator",  module_spec["estimator" ]["type"])(module_spec["estimator"  ]["spec"], self.model)
        self.sensors    = {}
        
        for i in range(len(module_spec["sensors"])):
            self.sensors[module_spec["sensors"][i]["spec"]["alias"]] = sensor.Sensor(module_spec["sensors"][i])

    def action(self, dt, sensors_data):
        u = sensors_data["control_sensor"]
        est_data, est_param = self.estimator.estimate(u,sensors_data)
        
        ret = {"control"  : []}
        
        if "communication_sensor" in self.sensors.keys():
            ret["broadcast"] = {
                "state":est_param["ego_state_est"]
            }

        return ret

    def get_state(self, dt, sensors_data):
        u = self.last_control
        est_data, est_param = self.estimator.estimate(u,sensors_data)
        
        state =  est_data['state_sensor_est']['state']

        return state





