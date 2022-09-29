import sys, os
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), '../../'))

import numpy as np
from abc import ABC, abstractmethod
from .FeedbackController import FeedbackController, NaiveFeedbackController
from .SafeController import SafeController, SafeSetController
from utils import GoalType

import importlib

class Controller(ABC):
    '''
        High-Level Controller
        - Directly called by agent
        - Invokes
            1. FeedbackController
            2. SafeController
    '''
    def __init__(self, spec, model):
        
        feedback_controller_spec = spec["feedback_controller"]
        safe_controller_spec     = spec["safe_controller"]
        self.feedback_controller = self._class_by_name("controller", feedback_controller_spec["type"])(feedback_controller_spec["spec"], model)
        self.safe_controller     = self._class_by_name("controller", safe_controller_spec["type"])(safe_controller_spec["spec"], model)

        self.model = model

    def _class_by_name(self, module_name, class_name):
        """Return the class handle by name of the class
        """
        module_name = "agent." + module_name
        ModuleClass = getattr(importlib.import_module(module_name), class_name)
        return ModuleClass

    def __call__(self,
        dt: float,
        processed_data: dict,
        goal: np.ndarray,
        goal_type: GoalType,
        state_dimension: int,
        external_action: np.ndarray = None) -> np.ndarray:
        
        if external_action is None:
            u_fb = self.feedback_controller(processed_data, goal, goal_type, state_dimension)
        else:
            u_fb = external_action

        # call safe controller
        u_safe, dphi = self.safe_controller(dt, processed_data, u_fb, goal, goal_type)

        return u_safe, dphi

        
