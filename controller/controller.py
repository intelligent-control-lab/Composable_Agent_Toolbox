import numpy as np
from abc import ABCMeta, abstractmethod
from .controller_manager import Controller_Manager

# Controller Base Class
class Controller_Base(object):
    '''
    This is the basic controller class template.
    One should complete the __init__(), control() and reset() function to instantiate a controller.
    '''
    __metaclass__ = ABCMeta

    _name = ''
    _type = '' # Coordination controller, Feedforward controller, Feedback controller, Safety controller ...

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

    @property
    def get_name(self):
        return self._name

    @property
    def get_type(self):
        return self._type


# A Blank Controller Class
class Zero_Controller(Controller_Base):
    '''
    This is a blank controller class.
    The Zero_Controller always returns 0 as controller output.
    '''
    def __init__(self, spec, model):
        self._name = 'Zero'
        self._type = 'None'
        self._spec = spec
        self._model = model

    def reset(self):
        return True

    def control(self, dt, x, goal_x, est_params):
        return np.zeros(self._model.u_shape)


# Controller Class
class Controller(Controller_Base):
    '''
    This is the unified controller class used in Agent Class.
    Controller specs should be given in the following format:
    {'coordination': 'CCC', 'feedback': 'PID', 'feedforward': 'Vel_FF', 'safety': 'SMA',
     'params': {'coordination': params, 'feedback': params, 'feedforward': params, 'safety': params}}
    The controller key can be ignored if it's not used, it will be filled with an zero controller automatically.

    E.g. a PID controller specs:
    {"feedback": "PID", "params": {"feedback": {"kp": [1, 1], "ki": [0, 0], "kd": [0, 0]}}}
    '''
    def __init__(self, spec, model):
        self._spec = spec
        self._model = model

        if 'coordination' in self._spec:
            self.coordination = globals()[self._spec['coordination']](self._spec['params']['coordination'], self._model)
        else:
            self.coordination = Zero_Controller(self._spec, self._model)
        
        if 'feedforward' in self._spec:
            self.feedforward = globals()[self._spec['feedforward']](self._spec['params']['feedforward'], self._model)
        else:
            self.feedforward = Zero_Controller(self._spec, self._model)

        if 'feedback' in self._spec:
            self.feedback = globals()[self._spec['feedback']](self._spec['params']['feedback'], self._model)
        else:
            self.feedback = Zero_Controller(self._spec, self._model)

        if 'safety' in self._spec:
            self.safety = globals()[self._spec['safety']](self._spec['params']['safety'], self._model)
        else:
            self.safety = Zero_Controller(self._spec, self._model)
    
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
        setattr(self, item, Zero_Controller(self._spec, self._model))
        return True

    def remove_all(self):
        self.coordination = Zero_Controller(self._spec, self._model)
        self.feedforward = Zero_Controller(self._spec, self._model)
        self.feedback = Zero_Controller(self._spec, self._model)
        self.safety = Zero_Controller(self._spec, self._model)
        return True

    def control(self, dt, x, goal_x, est_params):
        controller_manager = Controller_Manager(self.coordination, self.feedforward, self.feedback, self.safety)
        return controller_manager.build_controller(dt, x, goal_x, est_params)


# A PID Controller Class Example
class PID(Controller_Base):
    '''
    A multi-channel PID controller.
    '''
    def __init__(self, params, model):
        self._name = 'PID'
        self._type = 'feedback'
        self._params = params
        self._model = model
        self._kp = self._params['kp']
        self._ki = self._params['ki']
        self._kd = self._params['kd']
        self._e      = np.zeros(self._model.u_shape)
        self._e_last = np.zeros(self._model.u_shape)
        self._sum_e  = np.zeros(self._model.u_shape)

    def reset(self):
        self._e      = np.zeros(self._model.u_shape)
        self._e_last = np.zeros(self._model.u_shape)
        self._sum_e  = np.zeros(self._model.u_shape)
        return True

    def param_tuning(self, params):
        self._kp = params['kp']
        self._ki = params['ki']
        self._kd = params['kd']
        return True

    def control(self, dt, x, goal_x, est_params):
        self._e = np.array(goal_x).reshape(len(goal_x), 1) - np.array(x).reshape(len(x), 1)
        self._sum_e += self._e * dt
        output = np.diag(self._kp) * self._e + np.diag(self._ki) * self._sum_e + np.diag(self._kd) * (self._e - self._e_last) / dt
        return output

    def get_error(self, x, goal_x):
        return np.array(goal_x).reshape(len(goal_x), 1) - np.array(x).reshape(len(x), 1)


# Velocity Feedforward Controller Class
class Vel_FF(Controller_Base):
    '''
    A multi-channel velocity feedforward controller.
    '''
    def __init__(self, params, model):
        self._name = 'Vel_FF'
        self._type = 'feedforward'
        self._params = params
        self._model = model
        self._kv = self._params['kv']
        self._goal_x_last = 0

    def reset(self):
        return True

    def control(self, dt, x, goal_x, est_params):
        output = np.diag(self._kv) * (goal_x - self._goal_x_last) / dt
        self._goal_x_last = goal_x
        return output