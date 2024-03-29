from distutils.log import debug
import time

from matplotlib import pyplot as plt
from agent import sensor
import numpy as np
import importlib
from .agent_base import AgentBase

class ModelBasedAgent(AgentBase):
    def __init__(self, module_spec):
        self.instantiate_by_spec(module_spec)
        nu = (module_spec["model"]["control"]["spec"]["control_input_dim"])
        self.replanning_timer = self.planner.replanning_cycle
        self.last_control     = np.zeros((nu,1))  
        self.control_log = {}
        self.control_log['x'] = []
        self.control_log['y'] = []
        self.control_log['t'] = []
        self.start_time = time.time()
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

    def action(self, dt, sensors_data, debug_modes=None):

        # --------------------------- get previous control --------------------------- #
        u = self.last_control

        # ----------------------------- update estimation ---------------------------- #
        est_data, est_param = self.estimator.estimate(u,sensors_data)

        # ------------------------- update planned trajectory ------------------------ #
        goal = self.task.goal(est_data) # todo need goal type for planner
        if self.replanning_timer == self.planner.replanning_cycle:
            # add the future planning information for another agent 
            self.planned_traj = self.planner(dt, goal, est_data) # todo pass goal type
            self.replanning_timer = 0

        # --------------------------- select next waypoint --------------------------- #
        # next_traj_point = self.planned_traj[min(self.replanning_timer, self.planned_traj.shape[0]-1)]  # After the traj ran out, always use the last traj point for reference.
        # next_traj_point = np.vstack(next_traj_point.ravel())
        # print(min(self.replanning_timer, self.planned_traj.shape[0]-1))
        # next_traj_point = self.planner.next_point(
        #         self.planned_traj[min(self.replanning_timer, self.planned_traj.shape[0]-1):], est_data)
        next_traj_point = self.planner.next_point(self.planned_traj, est_data)
        self.replanning_timer += 1
        # if self.name == 'human':
        #     print(next_traj_point)
        # print(f'traj: {self.planned_traj}')

        # --------------------------- compute agent control -------------------------- #
        control = self.controller(
            dt, est_data, next_traj_point, self.task.goal_type(est_data),
            self.planner.state_dimension)
        
        self.last_control = control

        # self.control_log.append((control, time.time() - self.start_time))
        self.control_log['x'].append(control[0])
        self.control_log['y'].append(control[1])
        self.control_log['t'].append(time.time() - self.start_time)

        if debug_modes is not None:
            if debug_modes['log_traj']: 
                print(f'({self.name}) traj: {self.planned_traj}\n')
            if debug_modes['log_next_traj_point']: 
                print(f'({self.name}) next_traj_point: {next_traj_point}\n')
            if debug_modes['log_control']: 
                print(f'({self.name}) control: {control}\n')
            if debug_modes['plot_control'] and self.name == "human":
                self.plot_control()

        ret = {"control"  : control}
        if "communication_sensor" in self.sensors.keys():
            ret["broadcast"] = {
                # "planned_traj":self.planned_traj[min(self.replanning_timer, self.planner.horizon-1):],
                "planned_traj": self.planned_traj,
                "state":est_param["ego_state_est"],
                "next_point":next_traj_point
            }

        return ret

    def plot_control(self):
        if len(self.control_log['x']) == 1:
            plt.figure()
            self.ax = plt.axes(projection='3d')
        x = np.array(self.control_log['x']).flatten()
        y = np.array(self.control_log['y']).flatten()
        t = np.array(self.control_log['t']).flatten()
        self.ax.plot(x, y, t, color='blue')
        plt.pause(0.001)

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





