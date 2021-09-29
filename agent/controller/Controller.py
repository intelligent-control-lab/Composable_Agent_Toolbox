import sys, os
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), '../../'))

import numpy as np
from abc import ABC, abstractmethod
from .FeedbackController import FeedbackController, NaiveFeedbackController
from .SafeController import SafeController, SafeSetController
from utils import GoalType

class Controller(ABC):
    '''
        High-Level Controller
        - Directly called by agent
        - Invokes
            1. FeedbackController
            2. SafeController
    '''

    def __init__(self,
        feedback_controller:    FeedbackController,
        safe_controller:        SafeController):
        
        self.feedback_controller = feedback_controller
        self.safe_controller     = safe_controller

    def __call__(self,
        dt: float,
        processed_data: dict,
        goal: np.ndarray,
        goal_type: GoalType) -> np.ndarray:

        u = self.feedback_controller(processed_data, goal, goal_type)

        # call safe controller

        return u

        
