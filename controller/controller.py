import numpy as np
from abc import ABCMeta, abstractmethod

# Controller Base Class
class Controller_Base(object):
    '''
    This is the basic controller class template.
    One should complete the __init__(), control() and reset() function to instantiate a controller.
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
    def control(self):
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
    def __init__(self, spec, model):
        self.name = 'Zero'
        self.type = 'None'
        self.spec = spec
        self.model = model

    def reset(self):
        return True

    def control(self, dt, x, goal_x, est_params):
        return np.zeros(self.model.u_shape)


# Controller Class
class Controller(Controller_Base):
    '''
    This is the unified controller class used in Agent Class.
    Controller specs should be given in the following format:
    {'coordination': CCC, 'feedback': 'PID', 'feedforward': 'Vel_FF', 'Safety': 'SMA', 'params': {params}}
    The controller key can be ignored if it's not used, it will be filled with an zero controller automatically.

    E.g. a PID controller specs:
    {"feedback": "PID", "params": {"kp": [1, 1], "ki": [0, 0], "kd": [0, 0]}}
    '''    
    def __init__(self, spec, model):
        self.spec = spec
        self.model = model

        if 'coordination' in self.spec:
            self.coordination = globals()[self.spec['coordination']](self.spec['params'], self.model)
        else:
            self.coordination = Zero_Controller(self.spec, self.model)
        
        if 'feedforward' in self.spec:
            self.feedforward = globals()[self.spec['feedforward']](self.spec['params'], self.model)
        else:
            self.feedforward = Zero_Controller(self.spec, self.model)

        if 'feedback' in self.spec:
            self.feedback = globals()[self.spec['feedback']](self.spec['params'], self.model)
        else:
            self.feedback = Zero_Controller(self.spec, self.model)

        if 'Safety' in self.spec:
            self.safety = globals()[self.spec['safety']](self.spec['params'], self.model)
        else:
            self.safety = Zero_Controller(self.spec, self.model)
    
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
        setattr(self, item, Zero_Controller(self.spec, self.model))
        return True

    def remove_all(self):
        self.coordination = Zero_Controller(self.spec, self.model)
        self.feedforward = Zero_Controller(self.spec, self.model)
        self.feedback = Zero_Controller(self.spec, self.model)
        self.safety = Zero_Controller(self.spec, self.model)
        return True

    def control(self, dt, x, goal_x, est_params):
        coordination_output = self.coordination.control(dt, x, goal_x, est_params)
        feedforward_output  = self.feedforward.control(dt, x, goal_x, est_params)
        feedback_output     = self.feedback.control(dt, x, goal_x, est_params)
        safety_output       = self.safety.control(dt, x, goal_x, est_params)
        print(coordination_output)
        print(feedforward_output)
        print(feedback_output)
        print(safety_output)
        return coordination_output + feedforward_output + feedback_output + safety_output


# A PID Controller Class Example
class PID(Controller_Base):
    '''
    A multi-channel PID controller.
    '''
    def __init__(self, params, model):
        self.name = 'PID'
        self.params = params
        self.model = model
        self.kp = self.params['kp']
        self.ki = self.params['ki']
        self.kd = self.params['kd']
        self.e      = np.zeros(self.model.u_shape)
        self.e_last = np.zeros(self.model.u_shape)
        self.sum_e  = np.zeros(self.model.u_shape)

    def reset(self):
        self.e      = np.zeros(self.model.u_shape)
        self.e_last = np.zeros(self.model.u_shape)
        self.sum_e  = np.zeros(self.model.u_shape)
        return True

    def param_tuning(self, params):
        self.kp = params['kp']
        self.ki = params['ki']
        self.kd = params['kd']
        return True

    def control(self, dt, x, goal_x, est_params):
        print(x)
        print(goal_x)
        self.e = np.array(goal_x).reshape(len(goal_x), 1) - np.array(x).reshape(len(x), 1)
        self.sum_e += self.e * dt
        output = np.diag(self.kp) * self.e + np.diag(self.ki) * self.sum_e + np.diag(self.kd) * (self.e - self.e_last) / dt
        return output

    def get_error(self, x, goal_x):
        return np.array(goal_x).reshape(len(goal_x), 1) - np.array(x).reshape(len(x), 1)


# Velocity Feedforward Controller Class
class Vel_FF(Controller_Base):
    '''
    A multi-channel velocity feedforward controller.
    '''
    def __init__(self, params, model):
        self.name = 'PID'
        self.params = params
        self.model = model
        self.kv = self.params['kv']
        self.goal_x_last = 0

    def reset(self):
        return True

    def control(self, dt, x, goal_x, est_params):
        output = np.diag(self.kv) * (goal_x - self.goal_x_last) / dt
        print(output)
        return output