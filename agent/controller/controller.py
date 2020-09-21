import numpy as np
from abc import ABC, abstractmethod
from .controller_manager import Controller_Manager
import control # Python Control System Library

# Controller Base Class
class Controller_Base(ABC):
    '''
    This is the basic controller class template.
    One should complete the __init__(), control() and reset() function to instantiate a controller.

    Attributes:
        _name
        _type
    '''

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

    Attributes:
        _name
        _type
        _spec
        _model
    '''
    def __init__(self, spec, model):
        self._name = 'Zero'
        self._type = 'None'
        self._spec = spec
        self._model = model

    def reset(self):
        return True

    def control(self, dt, x, goal_x, est_params):
        return np.zeros(self._model.shape_u)


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

    Attributes:
        _name
        _type
        _spec
        _model
        coordination
        feedforward
        feedback
        safety
    '''
    def __init__(self, spec, model):
        self._name = 'Controller'
        self._type = 'Synthetic'
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
        
        # build the controller manager depends on "controller_manager.py"
        self._controller_manager = Controller_Manager(self.coordination, self.feedforward, self.feedback, self.safety)
    
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
        return self._controller_manager.build_controller(dt, x, goal_x, est_params)


# A PID Controller Class Example
class PID(Controller_Base):
    '''
    A multi-channel PID controller.

    Attributes:
        _name
        _type
        _params
        _model
        _kp
        _ki
        _kd
        _e
        _e_last
        _sum_e
    '''
    def __init__(self, params, model):
        self._name = 'PID'
        self._type = 'feedback'
        self._params = params
        self._model = model
        self._kp = self._params['kp']
        self._ki = self._params['ki']
        self._kd = self._params['kd']
        self._e      = np.zeros(self._model.shape_u)
        self._e_last = np.zeros(self._model.shape_u)
        self._sum_e  = np.zeros(self._model.shape_u)

    def reset(self):
        self._e      = np.zeros(self._model.shape_u)
        self._e_last = np.zeros(self._model.shape_u)
        self._sum_e  = np.zeros(self._model.shape_u)
        return True

    def param_tuning(self, params):
        self._kp = params['kp']
        self._ki = params['ki']
        self._kd = params['kd']
        return True

    def control(self, dt, x, goal_x, est_params):
        self._e = np.vstack(goal_x) - np.vstack(x)
        self._sum_e += self._e * dt
        output = np.diag(self._kp) @ self._e + np.diag(self._ki) @ self._sum_e + np.diag(self._kd) @ (self._e - self._e_last) / dt
        return output

    def get_error(self, x, goal_x):
        return np.array(goal_x).reshape(len(goal_x), 1) - np.array(x).reshape(len(x), 1)


# Velocity Feedforward Controller Class
class Vel_FF(Controller_Base):
    '''
    A multi-channel velocity feedforward controller.

    Attributes:
        _name
        _type
        _params
        _model
        _kv
        _goal_x_last
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
        output = np.diag(self._kv) @ (goal_x - self._goal_x_last) / dt
        self._goal_x_last = goal_x
        return output


# LQR Controller Class
class LQR(Controller_Base):
    '''
    An infinite-horizon, continuous-time LQR LQR controller.
    For a continuous time system, the state-feedback law u = -Kx minimizes the quadratic cost function
        J(u) = ∫_0^∞(x^TQx+u^TRu+2x^TNu)dt
    subject to the system dynamics
        dx/dt = Ax + Bu

    Attributes:
        _name
        _type
        _params
        _model
        _Q
        _R
        *_N
        _K
    '''
    def __init__(self, params, model):
        self._name = 'LQR'
        self._type = 'feedback'
        self._params = params
        self._model = model
        if 'Q' in self._params:
            self._Q = self._params['Q']
        else:
            raise Exception('No value of Q in params')
        
        if 'R' in self._params:
            self._R = self._params['R']
        else:
            raise Exception('No value of R in params')

        if 'N' in self._params:
            self._N = self._params['N']

    def reset(self):
        self._Q = np.zeros(self._model.A.shape)
        self._R = np.zeros((self._model.B.shape[1], self._model.B.shape[1]))
        self._K = np.zeros((self._model.B.shape[1], self._model.A.shape[0]))
        if hasattr(self, '_N'):
            self._N = np.zeros((self._model.A.shape[0], self._model.B.shape[1]))
        
        return True

    def param_tuning(self, params):
        if 'Q' in params:
            self._Q = params['Q']
        else:
            raise Exception('No value of Q in params')
        
        if 'R' in params:
            self._R = params['R']
        else:
            raise Exception('No value of R in params')

        if 'N' in params:
            self._N = params['N']
        
        return True

    def control(self, dt, x, goal_x, est_params):
        if hasattr(self, '_N'):
            K, S, E = control.lqr(self._model.A, self._model.B, self._Q, self._R, self._N)
        else:
            K, S, E = control.lqr(self._model.A, self._model.B, self._Q, self._R)
        
        self._K = K        
        output = - self._K @ x
        return output
        
