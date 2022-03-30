import time
from agent import sensor
import numpy as np
import importlib
from .agent_base import AgentBase

class FlatEvadeAgentMP(AgentBase):
    def __init__(self, module_spec):
        self.instantiate_by_spec(module_spec)
        nu = (module_spec["model"]["control"]["spec"]["control_input_dim"])
        self.replanning_timer = self.planner.replanning_cycle
        self.last_control     = np.zeros((nu,1))
        self.last_time = 0
        self.last_cycle = 0
        self.cycle_time = module_spec["model"]["planning"]["spec"]["cycle_time"]
        # the size of last_control should be equal to the number of control inputs, 
        # for flat_world with discs, each robot has two inputs, for franka arm, we assume direct control over end-eff, so 
        # the control input is 3D
        
    def _class_by_name(self, module_name, class_name):
        """Return the class handle by name of the class
        """
        module_name = "agent." + module_name
        ModuleClass = getattr(importlib.import_module(module_name), class_name)
        return ModuleClass

    def instantiate_by_spec(self, module_spec):
        """Instantiate modules based on user given specs
        """
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

    def init_action(self, sensor_data):
        
        u = self.last_control
        est_data, est_param = self.estimator.estimate(u, sensor_data[self.name])
        dt = sensor_data['time'] - self.last_time
        self.last_time += dt
        
        goal = self.task.goal(est_data) # todo need goal type for planner
        if self.replanning_timer == self.planner.replanning_cycle:
            # add the future planning information for another agent 
            self.planned_traj = self.planner(dt, goal, est_data) # todo pass goal type
            self.replanning_timer = 0

        next_traj_point = self.planned_traj[min(self.replanning_timer, self.planned_traj.shape[0]-1)]  # After the traj ran out, always use the last traj point for reference.
        next_traj_point = np.vstack(next_traj_point.ravel())
        self.replanning_timer += 1

        control = self.controller(
                dt, est_data, next_traj_point, self.task.goal_type(est_data),
                self.planner.state_dimension)
            
        self.last_control = control

        ret = {"control"  : control, "dt" : dt}
        if "communication_sensor" in self.sensors.keys():
            ret["broadcast"] = {
                "planned_traj":self.planned_traj[min(self.replanning_timer, self.planner.horizon-1):],
                "state":est_param["ego_state_est"]
            }
        
        return ret

    def action(self, mgr_actions, mgr_sensor_data, lock, iters):

        i = 0
        while i < iters:
            # ------------- compute dt and check user-specified cycle time -------------- #
            with lock:
                dt = mgr_sensor_data['time'] - self.last_time
            self.last_time += dt
            self.last_cycle += dt
            if self.last_cycle < self.cycle_time:
                time.sleep(0.001)
                continue
            i += 1
            self.last_cycle = 0
            print(f"agent {i}")

            # --------------------------- get previous control --------------------------- #
            u = self.last_control

            # ----------------------------- update estimation ---------------------------- #
            with lock:
                sensor_data = mgr_sensor_data[self.name]
            est_data, est_param = self.estimator.estimate(u, sensor_data)

            # ------------------------- update planned trajectory ------------------------ #
            goal = self.task.goal(est_data) # todo need goal type for planner
            if self.replanning_timer == self.planner.replanning_cycle:
                # add the future planning information for another agent 
                self.planned_traj = self.planner(dt, goal, est_data) # todo pass goal type
                self.replanning_timer = 0

            next_traj_point = self.planned_traj[min(self.replanning_timer, self.planned_traj.shape[0]-1)]  # After the traj ran out, always use the last traj point for reference.
            next_traj_point = np.vstack(next_traj_point.ravel())
            self.replanning_timer += 1

            # --------------------------- compute agent control -------------------------- #
            control = self.controller(
                dt, est_data, next_traj_point, self.task.goal_type(est_data),
                self.planner.state_dimension)
            
            self.last_control = control

            actions = {"control"  : control, "dt" : dt}
            if "communication_sensor" in self.sensors.keys():
                actions["broadcast"] = {
                    "planned_traj":self.planned_traj[min(self.replanning_timer, self.planner.horizon-1):],
                    "state":est_param["ego_state_est"]
                }

            with lock:
                mgr_actions[self.name] = actions



class UserControlAgent(AgentBase):
    def __init__(self, module_spec):
        self.instantiate_by_spec(module_spec)
        nu = (module_spec["model"]["spec"]["control_input_dim"])
        self.last_control     = np.zeros((nu,1))  
        
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

    def action(self, dt, sensor_data):
        u = sensor_data["control_sensor"]
        est_data, est_param = self.estimator.estimate(u,sensor_data)
        
        ret = {"control"  : []}
        
        if "communication_sensor" in self.sensors.keys():
            ret["broadcast"] = {
                "state":est_param["ego_state_est"]
            }

        return ret

    def get_state(self, dt, sensor_data):
        u = self.last_control
        est_data, est_param = self.estimator.estimate(u,sensor_data)
        
        state =  est_data['state_sensor_est']['state']

        return state





