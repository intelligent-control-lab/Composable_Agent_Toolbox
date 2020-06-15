import numpy as np
from abc import ABCMeta, abstractmethod

# Controller Base Class
class Controller_Base(object):
    '''
    This is the basic controller class template.
    One should complete the __init__(), get_control() and reset() function to instantiate a controller.
    '''
    __metaclass__ = ABCMeta

    controller_name = ''
    controller_type = '' # Coordination controller, Feedforward controller, Feedback controller, Safety controller ...

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        '''
        return TRUE for success, FALSE for failure
        '''
        pass

    @abstractmethod
    def get_control(self):
        '''
        This function calculate the controller output at each time step
        '''
        pass

    def get_name(self):
        return self.controller_name

    def get_type(self):
        return self.controller_type


# A Blank Controller Class
class Zero_Controller(Controller_Base):
    '''
    This is a blank controller class.
    The Zero_Controller always returns 0 as controller output.
    '''
    def __init__(self):
        self.name = 'Zero'
        self.type = 'None'

    def reset(self):
        return True

    def get_control(self, dt, x, goal_x, model):
        return 0


# Controller Class
class Controller(Controller_Base):
    '''
    This is the unified controller class used in Agent Class.
    Controller specs should be given in the following format:
    {'coordination': CCC, 'feedback': 'PID', 'feedforward': 'Vel_FF', 'Safety': 'SMA', 'params': {params}}
    The controller key can be ignored if it's not used, it will be filled with an zero controller automatically.

    E.g. a PID controller specs:
    {"feedback": "PID", "params": {"kp": 0, "ki": 0, "kd": 0}}
    '''
    def __init__(self, spec, model):
        if 'coordination' in spec:
            self.coordination = globals()[spec['coordination']](spec['params'])
        else:
            self.coordination = Zero_Controller()
        
        if 'feedforward' in spec:
            self.feedforward = globals()[spec['feedforward']](spec['params'])
        else:
            self.feedforward = Zero_Controller()

        if 'feedback' in spec:
            self.feedback = globals()[spec['feedback']](spec['params'])
        else:
            self.feedback = Zero_Controller()

        if 'Safety' in spec:
            self.safety = globals()[spec['safety']](spec['params'])
        else:
            self.safety = Zero_Controller()
    
    def reset(self, item):
        '''
        Reset specific module, return TRUE for success, FALSE for failure
        '''
        status = getattr(getattr(self, item), 'reset')()
        return status

    def reset_all(self):
        '''
        Reset ALL module, return TRUE for success, FALSE for failure
        '''
        stat1 = self.coordination.reset()
        stat2 = self.feedforward.reset()
        stat3 = self.feedback.reset()
        stat4 = self.safety.reset()
        return (stat1 and stat2 and stat3 and stat4)

    def remove(self, item):
        setattr(self, item, Zero_Controller())
        return True

    def remove_all(self):
        self.coordination = Zero_Controller()
        self.feedforward = Zero_Controller()
        self.feedback = Zero_Controller()
        self.safety = Zero_Controller()
        return True

    def get_control(self, dt, x, goal_x, model):
        coordination_output = self.coordination.get_control(dt, x, goal_x, model)
        feedforward_output  = self.feedforward.get_control(dt, x, goal_x, model)
        feedback_output     = self.feedback.get_control(dt, x, goal_x, model)
        safety_output       = self.safety.get_control(dt, x, goal_x, model)
        return coordination_output + feedforward_output + feedback_output + safety_output


# A PID Controller Class Example
class PID(Controller_Base):
    '''
    An one-channel PID controller.
    '''
    kp, ki, kd = 0.0, 0.0, 0.0
    e, e_last, sum_e = 0.0, 0.0, 0.0

    def __init__(self, spec):
        self.name = 'PID'
        self.kp = spec['kp']
        self.ki = spec['ki']
        self.kd = spec['kd']
        self.e_last, self.sum_e = 0.0, 0.0

    def reset(self):
        self.e, self.e_last, self.sum_e = 0.0, 0.0, 0.0
        return True

    def param_tuning(self, param):
        self.kp = param.kp
        self.ki = param.ki
        self.kd = param.kd
        return True

    def get_control(self, dt, x, goal_x, model):
        self.e = goal_x - x
        self.sum_e += self.e * dt
        output = self.kp * self.e + self.ki * self.sum_e + self.kd * (self.e - self.e_last) / dt
        return output

    def get_error(self, x, goal_x):
        return goal_x - x