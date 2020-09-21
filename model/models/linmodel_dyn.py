# Declare imports
# Style guide is currently following PEP8 - which means sort alphabetically

# Standard Library Import
from abc import ABC, abstractmethod  # This is needed for the abstract classes

# Third-party Imports
import numpy as np
import sympy as sp
import dill

# Application Specific Imports
from model.models.models import ModelBase

class LinearModel(ModelBase):
    '''
    class for Linear Models of form x_dot = Ax + Bu
    '''
    def __init__(self, spec):
        '''
        class constructor for Linear model class.

        This class constructor takes a spec that specifies the following options:
            use_spec    - exp: boolean - whether to leverage a spec defintion
            use_library - exp: boolean - whether to leverage dynamics library
            model_name  - exp: string - name of the model (in dynamics library)
            time_sample - exp: double/float - time sample value
            disc_flag   - exp: boolean - discrete time flag modifier
        '''

        # Flag for denoting hand-defined dynamics
        self.use_spec    = spec["use_spec"]
        self.use_library = spec["use_library"]
        self.model_name = spec["model_name"]
        self.time_sample = spec["time_sample"]
        self.disc_flag = spec["disc_flag"]
        self.symbol_dict = {} # Define an empty dictionary for holding the variables

        if self.use_spec:
            if self.use_library:
                model_lib_file = open(spec["model_spec"], 'rb')
                model_library = dill.load(model_lib_file)
                sys = model_library[spec["model_name"]]
                state_dec = self._declare_state(sys)
                cntl_dec = self._declare_cntl(sys)
                params_dec = self._declare_params(sys)

            else:
                # Define parameters based from passed specs
                state_dec = self._declare_state(spec["model_spec"]["syms"])
                # Declare Controls
                cntl_dec = self._declare_cntl(spec["model_spec"]["syms"])
                # Declare Parameters
                params_dec = self._declare_params(spec["model_spec"]["syms"])
        else:
        
            # Initialize States
            state_dec = self._declare_manual_state()
            self.x = state_dec[0] # This defines the state vector and gives it to the model object
            self.shape_x = state_dec[1] # This is the shape of the state vector
            self.state_dict = state_dec[2] # This is the state dictionary
            self.symbol_dict.update(state_dec[2]) # This updates the symbol dictionary to include the state symbol to mapping information

            # Declare Controls
            cntl_dec = self._declare_manual_cntl()
            self.u = cntl_dec[0] # This defines the control vector
            self.shape_u = cntl_dec[1] # This is the shape of the control vector
            self.cntl_dict = cntl_dec[2] # This is the control dictionary
            self.symbol_dict.update(cntl_dec[2]) # This updates the symbol dictionary to include the control symbol to name mapping information

            # Declare Parameters
            params_dec = self._declare_manual_params()
            self.params = params_dec[0] # This defines the parameter vector
            self.shape_params = params_dec[1] # This is the shape of the parameter vector
            self.param_dict = params_dec[2] # This is the parameter dictionary
            self.symbol_dict.update(params_dec[2]) # This updates the symbol dictionary to include the parameter symbol to name mapping information

            # Generate a list of all in use symbols
            self.all_syms = [[self.x, self.u, self.params]] # "Vertically stack" the states, parameter, control (really a three element list)

        # Set up all the model functions:
        if self.use_spec:
            if self.use_library:
                # Perform a look-up in the dynamics library
                self.cont_model = sys["Model_CT"]
                self.measure_func = sys["Measure Function"]
            else:
                # Define parameters based from passed specs
                self.A = self._declare_func_A(spec["model_spec"]["funcs"])
                self.B = self._declare_func_B(spec["model_spec"]["funcs"])
                self.measure_func = self._declare_func_measure(spec["model_spec"]["funcs"])
                self.cont_model = self.A*self.x + self.B*self.u
        else:
            # Currently, this assumes a system of the form: xdot = A*x + B*u
            self.A = self._declare_manual_func_A() 
            self.B = self._declare_manual_func_B()
            self.cont_model = self.A*self.x + self.B*self.u
            self.measure_func = self._declare_manual_func_measure() # Defines y = C*x - Currently only supports Full State Feedback

            if self.disc_flag:  # Discretize the System
                self.disc_model = super()._discretize_dyn(self.cont_model) # Discretizes the dynamics with a forward-Euler approximation
                self.cont_model_lam = super()._convert_funcs2lam(self.cont_model) # Converts the dynamics equations into a Python Lambda function (anonymous function)
                self.disc_model_lam = super()._convert_funcs2lam(self.disc_model) # Converts the dynamics equations into a Python Lambda function (anonymous function)
                self.measure_func_lam = super()._convert_funcs2lam(self.measure_func) # Converts the measurement equations into a Python Lambda function (anonymous function)
            else:
                self.cont_model_lam = super()._convert_funcs2lam(self.cont_model) # Converts the dynamics equations into a Python Lambda function (anonymous function)
                self.measure_func_lam = super()._convert_funcs2lam(self.measure_func) # Converts the measurement equations into a Python Lambda function (anonymous function)

    def _declare_manual_state(self):
        '''
        declare model states in SymPy Notation if not using dynamics library
        '''

        # Example placeholder is for a system with four states
        x1, x2, x3, x4 = sp.symbols(['x1', 'x2', 'x3', 'x4'])

        # Declare the states in aggregated vector form:
        x = sp.Matrix([x1, x2, x3, x4])
        shape_x = x.shape # Get the shape of x and return it

        # Declare your states with given names
        state_dict = {x1:'x position', x2:'x velocity', x3: 'y position', x4: 'y velocity'}

        # return the states in aggregation and the shape of the state vector:
        return [x, shape_x, state_dict]

    def _declare_manual_cntl(self):
        '''
        declare model control in SymPy Notation if not using dynamics library
        '''

        # Example placeholder is for a system with one control variable
        u1, u2 = sp.symbols(['u1', 'u2'])

        # Declare the controls in aggregated vector form:
        u = sp.Matrix([u1, u2])
        shape_u = u.shape # Get the shape of x and return it

        # Declare your controls with given names
        # cntl_dict = {u1:'torque', 'dumb_key':'torque2'}
        cntl_dict = {u1:'drive force in x', u2: 'drive force in y'}

        # return the controls in aggregation and the shape of the control vector:
        return [u, shape_u, cntl_dict]

    def _declare_manual_params(self):
        '''
        declare all model parameters in SymPy Notation if not using dynamics library
        '''

        # Example pacehold is for a system with three parameters
        m = sp.symbols('m')

        # Declare the parameters in aggregate vector form
        params = sp.Matrix([m])
        shape_params = params.shape # Get the parameter vector dimension

        # Declare your parameters with given names:
        param_dict = {m:'mass'}

        # return the parameters in aggregation and the shape.
        return [params, shape_params, param_dict]

    def _declare_manual_func_A(self):
        '''
        declare all the model dynamics function regularizaiton information using the SymPy notation
        '''

        # This is expecting a user-defined input, so if you have a different model, you will need to change this
        A = sp.Matrix([[0,1,0,0], [0,0,0,0], [0,0,0,1],[0,0,0,0]])
 
        return A

    def _declare_manual_func_B(self):
        '''
        declare all the model dynamics function control weight information using the SymPy notation
        '''

        # This is expecting a user-defined input, so if you have a different model, you will need to change this
        B = sp.Matrix([[0,0], [1/self.params[0], 0], [0,0],[0,1/self.params[0]]])

        return B

    def _declare_manual_func_measure(self):
        '''
        declare all measurement outputs for the system
        '''

        # Currently, this returns a full-state feedback system, future implementations will likely require different measurement functions
        return sp.eye(self.shape_x[0])*self.x

   # Opposed to the previous three functions, these take in the arguments from the specification:
    def _declare_func_f(self, func_dict):
        '''
        declare all the model dynamics function regularizaiton information using the SymPy notation
        '''
        A_expr = func_dict["A"] # Find "f_expr" function value
        return A_expr

    def _declare_func_B(self, func_dict):
        '''
        declare all the model dynamics in the specificiation
        '''
        B_expr = func_dict["B"]

        return B_expr

    def _declare_func_measure(self, func_dict):
        '''
        declare all measurement outputs for the system
        '''
        m_func = func_dict["m_func"]
        return m_func

    def evaluate_dynamics(self, x_sub, u_sub, params_sub):
        '''
        provide an interface for use with the superclass
        evaluate_dynamics function
        '''
        return ModelBase.evaluate_dynamics(self, x_sub, u_sub, params_sub)

    def evaluate_measurement(self, x_sub):
        '''
        provide an interface for use with the superclass
        evaluate_dynamics function
        '''
        return ModelBase.evaluate_measurement(self, x_sub)