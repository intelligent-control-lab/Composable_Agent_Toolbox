import numpy as np
from abc import ABC, abstractmethod
import cvxopt

# Controller Base Class
class Controller_Base(ABC):
    """
    This is the basic controller class template.
    One should complete the __init__(), control() and reset() function to instantiate a controller.

    Attributes:
        _name
        _type
    """

    _name = ''
    _type = '' # Coordination controller, Feedforward controller, Feedback controller, Safety controller ...

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        """
        return TRUE for success, FALSE for failure
        """
        pass

    @abstractmethod
    def control(self):
        """
        This function calculate the controller output at each time step
        """
        pass

    @property
    def get_name(self):
        return self._name

    @property
    def get_type(self):
        return self._type



# 
class BaseSafeController(Controller_Base):
    """
    This is a blank controller class.
    The Zero_Controller always returns 0 as controller output.

    Attributes:
        _name
        _type
        _spec
        _model
    """
    def __init__(self, spec, model):
        self._type = 'safety'
        self._spec = spec
        self._model = model
        self.d_min = spec['d_min']
        self.eta = spec['eta']
        self.k_v = spec['k_v']
        
    def reset(self):
        return True

    def phi_and_derivatives(self, dt, ce, co):
        n = np.shape(ce)[0]//2

        dp = ce[:n] - co[:n]
        dv = ce[n:] - co[n:]

        d     = max(np.linalg.norm(dp), 1e-3)
        dot_d = max(np.linalg.norm(dv), 1e-3)

        phi = self.d_min**2 - d**2 - self.k_v * dot_d + self.eta * dt

        p_phi_p_d = -2 * d
        p_phi_p_dot_d = - self.k_v

        p_d_p_ce = np.vstack([dp / d, np.zeros((n,1))])
        p_d_p_co = -p_d_p_ce

        p_dot_d_p_ce = np.vstack([np.zeros((n,1)), dv / d])
        p_dot_d_p_co = -p_dot_d_p_ce

        p_phi_p_ce = p_phi_p_d * p_d_p_ce + p_phi_p_dot_d * p_dot_d_p_ce
        p_phi_p_co = p_phi_p_d * p_d_p_co + p_phi_p_dot_d * p_dot_d_p_co
        
        return phi, p_phi_p_ce, p_phi_p_co
    
    @abstractmethod
    def safe_control(self, u0, dt, obs, est_data, est_params):
        """ Compute the safe control between ego and an obstacle.
        """
        pass
    
    def control(self, u0, dt, x, goal_x, est_data, est_params):
        us = []
        for obs in est_data["obstacle_sensor"]:
            phi, u = self.safe_control(u0, dt, obs, est_data, est_params)
            us.append((phi, u))
        
        sorted(us, key=lambda x:x[0], reverse=True) # larger phi first
        
        return us[0] # adopt the control that avoids the most dangerous collision.


class SafeSetController(BaseSafeController):
    def __init__(self, spec, model):
        super().__init__(spec, model)
        self._name = 'safe_set'
        
    def safe_control(self, u0, dt, obs, est_data, est_params):
        """ Compute the safe control between ego and an obstacle.

        Safe set compute u by solving the following optimization:
        min || u - u0 ||, 
        s.t.  dot_phi < eta  or  phi > 0 (eta is the safety margin used in phi)

        => p_phi_p_xe.T * dot_xe        + p_phi_p_co.T * dot_co < eta
        => p_phi_p_xe.T * (fx + fu * u) + p_phi_p_co.T * dot_co < eta
        => p_phi_p_xe.T * fu * u < eta - p_phi_p_xe.T * fx - p_phi_p_co.T * dot_co

        """
        ce = np.vstack([est_data["cartesian_sensor"]["pos"], est_data["cartesian_sensor"]["vel"]])  # ce: cartesian state of ego
        co = np.vstack([est_data["obstacle_sensor"][obs]["rel_pos"], est_data["obstacle_sensor"][obs]["rel_vel"]]) + ce  # co: cartesian state of the obstacle
        
        n = np.shape(ce)[0]//2

        # It will be better if we have an estimation of the acceleration of the obstacle
        dot_co = np.vstack([co[n:], np.zeros((n,1))])

        phi, p_phi_p_ce, p_phi_p_co = self.phi_and_derivatives(dt, ce, co)

        p_ce_p_xe = self._model.p_ce_p_xe()
        fx = self._model.fx()
        fu = self._model.fu()

        p_phi_p_xe = p_ce_p_xe.T * p_phi_p_ce

        L = p_phi_p_xe.T * fu
        S = self.eta - p_phi_p_xe.T * fx - p_phi_p_co.T * dot_co
        
        u = u0

        if phi <= 0 or np.ndarray.item(L * u0) < np.ndarray.item(S):
            u = u0
        else:
            u = u0 - (np.ndarray.item(L * u0 - S) * L.T / np.ndarray.item(L * L.T))
        
        return phi, u


class PotentialFieldController(BaseSafeController):
    def __init__(self, spec, model):
        super().__init__(spec, model)
        self._name = 'potential_field'
        self.c = spec['c']
        
    def safe_control(self, u0, dt, obs, est_data, est_params):
        """ Compute the safe control between ego and an obstacle.

        Potential Field compute the safe control by first computing a safe control in the cartesian space,
        then compute the state space control by inverse kinematics.

        This method is not suitable for all control systems.
        
        u_cartesian = -c * p_phi_p_ce,
        u = fu.T * p_ce_p_xe.T * u_ce

        """
        ce = np.vstack([est_data["cartesian_sensor"]["pos"], est_data["cartesian_sensor"]["vel"]])  # ce: cartesian state of ego
        co = np.vstack([est_data["obstacle_sensor"][obs]["rel_pos"], est_data["obstacle_sensor"][obs]["rel_vel"]]) + ce  # co: cartesian state of the obstacle
        
        n = np.shape(ce)[0]//2

        # It will be better if we have an estimation of the acceleration of the obstacle
        dot_co = np.vstack([co[n:], np.zeros((n,1))])

        phi, p_phi_p_ce, p_phi_p_co = self.phi_and_derivatives(dt, ce, co)

        p_ce_p_xe = self._model.p_ce_p_xe()
        fx = self._model.fx()
        fu = self._model.fu()

        u_ce = -self.c * p_phi_p_ce

        u = fu.T * p_ce_p_xe.T * u_ce

        return phi, u


class ZeroingBarrierFunctionController(BaseSafeController):
    def __init__(self, spec, model):
        super().__init__(spec, model)
        self._name = 'zeroing_barrier_function'
        self.lambd = spec['lambd']
        
        
    def safe_control(self, u0, dt, obs, est_data, est_params):
        """ Compute the safe control between ego and an obstacle.

        Zeroing Barrier Function compute u by solving the following optimization:
        min || u - u0 ||, 
        s.t.  dot_phi < lambd * phi

        => p_phi_p_xe * dot_xe          + p_phi_p_co * dot_co < lambd * phi
        => p_phi_p_xe * ( fx + fu * u ) + p_phi_p_co * dot_co < lambd * phi
        => p_phi_p_xe * fu * u < lambd * phi - p_phi_p_xe * fx - p_phi_p_co * dot_co

        """
        ce = np.vstack([est_data["cartesian_sensor"]["pos"], est_data["cartesian_sensor"]["vel"]])  # ce: cartesian state of ego
        co = np.vstack([est_data["obstacle_sensor"][obs]["rel_pos"], est_data["obstacle_sensor"][obs]["rel_vel"]]) + ce  # co: cartesian state of the obstacle
        
        n = np.shape(ce)[0]//2

        # It will be better if we have an estimation of the acceleration of the obstacle
        dot_co = np.vstack([co[n:], np.zeros((n,1))])

        phi, p_phi_p_ce, p_phi_p_co = self.phi_and_derivatives(dt, ce, co)

        p_ce_p_xe = self._model.p_ce_p_xe()
        fx = self._model.fx()
        fu = self._model.fu()

        p_phi_p_xe = p_ce_p_xe.T * p_phi_p_ce

        A = cvxopt.matrix(p_phi_p_xe.T * fu)
        b = cvxopt.matrix(self.lambd * phi - p_phi_p_xe.T * fx - p_phi_p_co.T * dot_co)
        A = A / abs(b)
        b = b / abs(b)

        Q = cvxopt.matrix(np.eye(np.shape(u0)[0]))
        p = cvxopt.matrix(- 2 * u0)
        A = cvxopt.matrix([[A]])
        b = cvxopt.matrix([[b]])

        u = u0
        try:
            cvxopt.solvers.options['feastol']=1e-9
            cvxopt.solvers.options['show_progress'] = False
            sol=cvxopt.solvers.qp(Q, p, A, b)
            u = np.vstack(sol['x'])
        except:
            pass

        return phi, u



class SlidingModeController(BaseSafeController):
    def __init__(self, spec, model):
        super().__init__(spec, model)
        self._name = 'sliding_mode'
        self.c = spec['c']
        
    def safe_control(self, u0, dt, obs, est_data, est_params):
        """ Compute the safe control between ego and an obstacle.

        Zeroing Barrier Function compute u by solving the following optimization:

        u = u0 + c * p_phi_p_xe * fu  when  phi > 0

        c is a large constant

        """
        ce = np.vstack([est_data["cartesian_sensor"]["pos"], est_data["cartesian_sensor"]["vel"]])  # ce: cartesian state of ego
        co = np.vstack([est_data["obstacle_sensor"][obs]["rel_pos"], est_data["obstacle_sensor"][obs]["rel_vel"]]) + ce  # co: cartesian state of the obstacle
        
        n = np.shape(ce)[0]//2

        # It will be better if we have an estimation of the acceleration of the obstacle
        dot_co = np.vstack([co[n:], np.zeros((n,1))])

        phi, p_phi_p_ce, p_phi_p_co = self.phi_and_derivatives(dt, ce, co)

        p_ce_p_xe = self._model.p_ce_p_xe()
        fx = self._model.fx()
        fu = self._model.fu()

        p_phi_p_xe = p_ce_p_xe.T * p_phi_p_ce
        
        if phi > 0:
            u = u0 - self.c * p_phi_p_xe.T * fu
        else:
            u = u0        

        return phi, u


class SublevelSafeSetController(BaseSafeController):
    def __init__(self, spec, model):
        super().__init__(spec, model)
        self._name = 'sublevel_safe_set'
        self.lambd = spec['lambd']
        
    def safe_control(self, u0, dt, obs, est_data, est_params):
        """ Compute the safe control between ego and an obstacle.

        Safe set compute u by solving the following optimization:
        min || u - u0 ||, 
        s.t.  dot_phi < lambda * dot_phi  or  phi > 0

        => p_phi_p_xe * dot_xe          + p_phi_p_co * dot_co < lambd * phi
        => p_phi_p_xe * ( fx + fu * u ) + p_phi_p_co * dot_co < lambd * phi
        => p_phi_p_xe * fu * u < lambd * phi - p_phi_p_xe * fx - p_phi_p_co * dot_co

        """
        ce = np.vstack([est_data["cartesian_sensor"]["pos"], est_data["cartesian_sensor"]["vel"]])  # ce: cartesian state of ego
        co = np.vstack([est_data["obstacle_sensor"][obs]["rel_pos"], est_data["obstacle_sensor"][obs]["rel_vel"]]) + ce  # co: cartesian state of the obstacle
        
        n = np.shape(ce)[0]//2

        # It will be better if we have an estimation of the acceleration of the obstacle
        dot_co = np.vstack([co[n:], np.zeros((n,1))])

        phi, p_phi_p_ce, p_phi_p_co = self.phi_and_derivatives(dt, ce, co)

        p_ce_p_xe = self._model.p_ce_p_xe()
        fx = self._model.fx()
        fu = self._model.fu()

        p_phi_p_xe = p_ce_p_xe.T * p_phi_p_ce

        L = p_phi_p_xe.T * fu
        S = self.lambd * phi - p_phi_p_xe.T * fx - p_phi_p_co.T * dot_co
        
        u = u0

        if phi <= 0 or np.ndarray.item(L * u0) < np.ndarray.item(S):
            u = u0
        else:
            u = u0 - (np.ndarray.item(L * u0 - S) * L.T / np.ndarray.item(L * L.T))
        
        return phi, u
