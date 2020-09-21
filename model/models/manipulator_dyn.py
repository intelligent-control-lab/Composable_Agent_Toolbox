# Declare imports
# Style guide is currently following PEP8 - which means sort alphabetically

# Standard Library Import
from abc import ABC, abstractmethod  # This is needed for the abstract classes

# Third-party Imports
import numpy as np
import sympy as sp
import dill

# Application Specific Imports
from models import ModelBase

class ManipulatorEquation(ModelBase):
    '''
    class for nonlinear control affine models of form q_ddot = h(q)^(-1) * (u - c(q, q_dot)*q_dot - g(q)) where
    q is the joint angle, the dots refer to the time derivatives with # of d's refering to the relative degree
    '''
    def __init__(self, spec):
        '''
        class constructor for a nonlinear model class.

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
        self.model_name  = spec["model_name"]
        self.time_sample = spec["time_sample"]
        self.disc_flag   = spec["disc_flag"]
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
            # Declare States
            state_dec = self._declare_manual_state()
            # Declare Controls
            cntl_dec = self._declare_manual_cntl()
            # Declare Parameters
            params_dec = self._declare_manual_params()

        # Complete setting up the model
        # Initialize States
        self.x = state_dec[0] # This defines the state vector and gives it to the model object
        self.shape_x = state_dec[1] # This is the shape of the state vector
        self.state_dict = state_dec[2] # This is the state dictionary
        self.symbol_dict.update(state_dec[2]) # This updates the symbol dictionary to include the state symbol to mapping information

        # Initialize Controls
        self.u = cntl_dec[0] # This defines the control vector
        self.shape_u = cntl_dec[1] # This is the shape of the control vector
        self.cntl_dict = cntl_dec[2] # This is the control dictionary
        self.symbol_dict.update(cntl_dec[2]) # This updates the symbol dictionary to include the control symbol to name mapping information

        # Initialize Parameters
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
                self.h_expr = self._declare_func_h(spec["model_spec"]["funcs"])
                self.c_expr = self._declare_func_c(spec["model_spec"]["funcs"])
                self.g_expr = self._declare_func_g(spec["model_spec"]["funcs"])
                self.measure_func = self._declare_func_measure(spec["model_spec"]["funcs"])
                self.cont_model = self.h_expr.inv() * (self.u - self.c_expr*sp.Matrix([[self.x[2]],[self.x[3]]]) - self.g_expr)
        else:
            # Manually define the functional components of the dynamic sturcture
            # The expression that's defined here is: q_ddot = h(q)^(-1) * (u - c(q, q_dot)*q_dot - g(q))
            self.h_expr = self._declare_manual_func_h() # Defines h(q) 
            self.c_expr = self._declare_manual_func_c() # Defines c(q, q_dot)
            self.g_expr = self._declare_manual_func_g() # Defines g(q)
            # Defines y = C*x - Currently only supports Full State Feedback
            self.measure_func = self._declare_manual_func_measure()
            self.cont_model = self.h_expr.inv() * (self.u - self.c_expr*sp.Matrix([[self.x[2]],[self.x[3]]]) - self.g_expr)

        self.cont_model = sp.Matrix([[self.x[2]],[self.x[3]], self.cont_model])
        
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
        q1, q2, q1_dot, q2_dot = sp.symbols(['q1', 'q2', 'q1_dot', 'q2_dot'])

        # Declare the states in aggregated vector form:
        x = sp.Matrix([q1, q2, q1_dot, q2_dot])
        shape_x = x.shape # Get the shape of x and return it

        # Declare your states with given names
        state_dict = {q1:'joint 1 angle', q2:'joint 2 angle', q1_dot:'joint 1 angular velocity', q2_dot:'joint 2 angular velocity'}

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
        cntl_dict = {u1:'joint 1 torque', u2:'joint 2 torque'}

        # return the controls in aggregation and the shape of the control vector:
        return [u, shape_u, cntl_dict]

    def _declare_manual_params(self):
        '''
        declare all model parameters in SymPy Notation if not using dynamics library
        '''


        # Example pacehold is for a system with three parameters
        m1, l1, lc1, I1, m2, l2, lc2, I2, g = sp.symbols(['m1', 'l1', 'lc1', 'I1', 'm2', 'l2', 'lc2', 'I2', 'g'])

        # Declare the parameters in aggregate vector form
        params = sp.Matrix([m1, l1, lc1, I1, m2, l2, lc2, I2, g])
        shape_params = params.shape # Get the parameter vector dimension

        # Declare your parameters with given names:
        param_dict = {m1: 'mass of link 1', l1: 'total length of link 1', lc1: 'length to center of mass of link 1 from joint 1', I1: 'inertia of Link 1', m2: 'mass of link 2', l2: 'total length of link 2', lc2: 'length to center of mass of link 2 from joint 2', I2: 'inertial of link 2', g: 'acceleration due to gravity'}

        # return the parameters in aggregation and the shape.
        return [params, shape_params, param_dict]


    def _declare_manual_func_h(self):
        '''
        declare all the model dynamics function regularizaiton information using the SymPy notation
        '''
        H11 = self.params[0]*self.params[2]**2 + self.params[3] + self.params[4]*(self.params[1]**2 + self.params[6]**2 + 2*self.params[1]*self.params[6]*sp.cos(self.x[1])) + self.params[7]
        H12 = self.params[4]*self.params[1]*self.params[6]*sp.cos(self.x[1]) + self.params[4]*self.params[6]**2 + self.params[7]
        H22 = self.params[4]*self.params[6]**2 + self.params[7]
        # Define Inertial Matrix
        h_expr = sp.Matrix([[H11, H12], [H12, H22]])

        return h_expr

    def _declare_manual_func_c(self):
        '''
        declare all the model dynamics function regularizaiton information using the SymPy notation
        '''
        h = self.params[4]*self.params[1]*self.params[6]*sp.sin(self.x[1])
        C11 = -1*h*self.x[3]
        C12 = -1*h*self.x[2] + -1*h*self.x[3]
        C21 = h * self.x[2]
        C22 = 0
        # Define Inertial Matrix
        c_expr = sp.Matrix([[C11, C12], [C21, C22]])

        return c_expr

    def _declare_manual_func_g(self):
        '''
        declare all the model dynamics function control weight information using the SymPy notation
        '''
        G11 = self.params[0]*self.params[2]*self.params[8]*sp.cos(self.x[0]) + self.params[4]*self.params[8]*(self.params[6]*sp.cos(self.x[0] + self.x[1]) + self.params[1]*sp.cos(self.x[0]))
        G21 = self.params[4]*self.params[6]*self.params[8]*sp.cos(self.x[0] + self.x[1])
        # This is expecting a user-defined input
        g_expr = sp.Matrix([[G11], [G21]])

        return g_expr

    def _declare_manual_func_measure(self):
        '''
        declare all measurement outputs for the system
        '''

        # Currently, this returns a full-state feedback system, future implementations will likely require different measurement functions
        return sp.eye(self.shape_x[0])*self.x

    # Opposed to the previous three functions, these take in the arguments from the specification:
    def _declare_func_h(self, func_dict):
        '''
        declare all the model dynamics function regularizaiton information using the SymPy notation
        '''
        h_expr = func_dict["h_expr"] # Find "f_expr" function value
        return h_expr

    def _declare_func_c(self, func_dict):
        '''
        declare all the model dynamics function regularizaiton information using the SymPy notation
        '''
        c_expr = func_dict["c_expr"] # Find "c_expr" function value

        return c_expr

    def _declare_func_g(self, func_dict):
        '''
        declare all the model dynamics in the specificiation
        '''
        g_expr = func_dict["g_expr"] # Find the "g_expr" function value

        return g_expr

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