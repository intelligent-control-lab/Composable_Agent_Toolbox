import importlib
from agent import sensor

import numpy as np

class MASAgent():

    def __init__(self, module_spec: dict) -> None:
        self._instantiate_by_spec(module_spec)
        self.path = [np.array(self.module_spec["task"]["init_x"])]
        self.goal = {"goal": np.array(self.module_spec["task"]["goal"]).reshape(2, 2)}
        self.dt = 0.02
        self.at_goal = False
        nu = (module_spec["model"]["control"]["spec"]["control_input_dim"])
        self.last_control = np.zeros((nu,1))  

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

    def get_waypoint(self) -> np.array:
        data = {"cartesian_sensor_est": {"pos": self.path[-1][:2], "vel": self.path[-1][2:]}}
        plan = self.planner(self.dt, self.goal, data)
        return plan[1]

    def set_path(self, path: list[np.array]) -> None:
        self.path = path
        # print(f'\n{self.name} {self.path[-1][:2]} {self.goal["goal"]}')
        # TODO: clean this up later
        if np.linalg.norm(self.path[-1][:2] - self.goal['goal']) < 0.1:
            self.at_goal = True
        if np.linalg.norm(self.path[-1][:2] - self.goal['goal'][0]) < 0.1:
            self.at_goal = True
        
    def action(self, dt, sensor_data) -> None:

        u = self.last_control
        est_data, est_param = self.estimator.estimate(u, sensor_data)
        next_traj_point = self.planner.next_point(self.path, est_data)

        control = self.controller(
            dt, est_data, next_traj_point, self.task.goal_type(est_data),
            self.planner.state_dimension)
        self.last_control = control

        ret = {"control"  : control}
        if "communication_sensor" in self.sensors.keys():
            ret["broadcast"] = {
                "planned_traj": self.path,
                "state":est_param["ego_state_est"],
                "next_point":next_traj_point
            }
        return ret

