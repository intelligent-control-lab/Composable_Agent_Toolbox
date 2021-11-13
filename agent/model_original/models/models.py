# Declare imports
# Style guide is currently following PEP8 - which means sort alphabetically

# Standard Library Import
from abc import ABC, abstractmethod  # This is needed for the abstract classes

# Third-party Imports
import numpy as np
import sympy as sp
import dill

# Application Specific Imports
# None

class AbstractModelBase(ABC):
    '''
    define an base class for developing models

    Classes derived from the model base need to implement copies
    of all abstract methods, lest a TypeError arises.
    '''

    @abstractmethod
    def __init__(self):
        pass

    # These only need to be populated if you don't use the specification/library method
    @abstractmethod
    def _declare_manual_state(self):
        pass

    @abstractmethod
    def _declare_manual_cntl(self):
        pass

    @abstractmethod
    def _declare_manual_params(self):
        pass


class ModelBase(AbstractModelBase):
    '''
    split classes so that we have an abstract model base and regular ModelBase
    '''

    def _declare_state(self, sym_dict):
        '''
        declare model states in SymPy Notation if not using dynamics library
        '''

        # Declare the states in aggregated vector form:
        x = sp.Matrix(sym_dict["states"])
        shape_x = x.shape # Get the shape of x and return it

        # Store states with given names:
        state_dict = sym_dict["state_dict"]

        # return the states in aggregation and the shape of the state vector:
        return [x, shape_x, state_dict]

    def _declare_cntl(self, sym_dict):
        '''
        declare model control in SymPy Notation if not using dynamics library
        '''
        # Declare the controls in aggregated vector form:
        u = sp.Matrix(sym_dict["cntls"])
        shape_u = u.shape # Get the shape of x and return it

        # Declare your controls with given names
        # cntl_dict = {u1:'torque', 'dumb_key':'torque2'}
        cntl_dict = sym_dict["cntl_dict"]

        # return the controls in aggregation and the shape of the control vector:
        return [u, shape_u, cntl_dict]

    def _declare_params(self, sym_dict):
        '''
        declare all model parameters in SymPy Notation if not using dynamics library
        '''
        # Declare the parameters in aggregate vector form
        params = sp.Matrix(sym_dict["params"])
        shape_params = params.shape # Get the parameter vector dimension

        # Declare your parameters with given names:
        param_dict = sym_dict["param_dict"]

        # return the parameters in aggregation and the shape.
        return [params, shape_params, param_dict]

    def _discretize_dyn(self, func):
        '''
        use a first order Euler Approximation to discretize the system dynamics
        '''
        dis_model = self.x + self.time_sample*func
        return dis_model

    def _convert_funcs2lam(self, func):
        '''
        this method converts the functions to python lambda functions for use in numerical evaluation note that the reason why we are passing the argument func here is that we can specify whichever function we like
        '''

        # This parses through all the symbols and returns an anonymous function that takes a single iterable input of three elements, each also being a list of state, control, and parameters in that order
        lam_func = sp.lambdify(self.all_syms, func, 'numpy')
        return lam_func

    def _take_jacobian(self, func, vars):
        '''
        this method linearizes the system dynamics by taking the jacobian of func with respect to argument X and returns a symbolic object
        '''

        # Leverages the SymPy dynamics structure
        return func.jacobian(vars)

    def _sub_vals(self, func, sub_vect, sub_val):
        '''
        substitute the values given in sub_val for those in sub_vect in expression func.

        sub_vect is a symbolic vector that will be passed for substitution
        sub_val is list of the same size as sub_vect that has the vals we want to substitute
        '''

        # we use zip() to element to element match sub_vect and sub_val
        sub_dict = dict(zip(sub_vect, sub_val))
        return func.subs(sub_dict)


    def evaluate_dynamics(self, x_sub, u_sub, params_sub):
        '''
        evaluate the dynamics with a specific values substituted for the state and the parameters

        Particularly slow way of evaluating the dynamics, but it does work if you want the symbolic objects
        '''
        if self.disc_flag:
            evaled = self._sub_vals(self.disc_model, self.x, x_sub) # Symbolically Evaluate States
            evaled = self._sub_vals(evaled, self.u, u_sub) # Symbolically Evaluate Controls
            evaled = self._sub_vals(evaled, self.params, params_sub) # Symbolically Evaluate params
            # return a symbolic math object (for manipulation) as well as a casted numpy object
            return [evaled, np.array(evaled).astype(np.float64)]

        # Pythonic Else - Evaluate the expression using a continuous time flag
        evaled = self._sub_vals(self.cont_model, self.x, x_sub)
        evaled = self._sub_vals(evaled, self.u, u_sub)
        evaled = self._sub_vals(evaled, self.params, params_sub)
        # return a symbolic math object (for manipulation) as well as a casted numpy object
        return [evaled, np.array(evaled).astype(np.float64)]

    def evaluate_measurement(self, x_sub):
        '''
        evaluate the measurement function with a specific value substituted fro the state and parameters. Currently assumes full-state feedback
        '''
        evaled = self._sub_vals(self.measure_func, self.x, x_sub) # Symbolically Evaluate States
        return [evaled, np.array(evaled).astype(np.float64)] # return a symbolic math object (for symbolic manipulation) as well as a casted numpy object

    def _declare_manual_state(self):
        pass

    def _declare_manual_cntl(self):
        pass

    def _declare_manual_params(self):
        pass
