# Declare imports
# Style guide is currently following PEP8 - which means sort alphabetically

# Standard Library Import
from abc import ABC, abstractmethod  # This is needed for the abstract classes

# Third-party Imports
import numpy as np
import sympy as sp
import dill

# Application Specific Imports
from agent.model.models.models import ModelBase

class NonlinModelCntlAffine(ModelBase):
    '''
    class for nonlinear control affine models of form x_dot = f(x) + g(x)u
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
        self.lin_model = None # Only populate if we actually linearize
        self.lin_measure_model = None # Only populate if we actually linearize
        self.lin_model_lam = None # Only populate if we actually linearize
        self.lin_measure_model_lam = None # Only populate if we actually linearize
        self.jac_x_lam = None # Only populate if we actually linearize
        self.jac_u_lam = None # Only populate if we actually linearize
        self.jacobian_x = None # Only populate if we actually linearize
        self.jacobian_u = None # Only populate if we actually linearize

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
                self.f_expr = self._declare_func_f(spec["model_spec"]["funcs"])
                self.g_expr = self._declare_func_g(spec["model_spec"]["funcs"])
                self.measure_func = self._declare_func_measure(spec["model_spec"]["funcs"])
                self.cont_model = self.f_expr + self.g_expr*self.u # Actually form the Right Hand Side (RHS) xdot = f(x) + g(x)u
        else:
            # Manually define the functional components of the dynamic sturcture
            self.f_expr = self._declare_manual_func_f() # Defines f(x) in xdot = f(x) + g(x)u
            self.g_expr = self._declare_manual_func_g() # Defines g(x) in xdot = f(x) + g(x)u
            # Defines y = C*x - Currently only supports Full State Feedback
            self.measure_func = self._declare_manual_func_measure() 
            self.cont_model = self.f_expr + self.g_expr*self.u # Actually form the Right Hand Side (RHS) xdot = f(x) + g(x)u


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
        x1, x2 = sp.symbols(['x1', 'x2'])

        # Declare the states in aggregated vector form:
        x = sp.Matrix([x1, x2])
        shape_x = x.shape # Get the shape of x and return it

        # Declare your states with given names
        state_dict = {x1:'angle', x2:'angular velocity'}

        # return the states in aggregation and the shape of the state vector:
        return [x, shape_x, state_dict]

    def _declare_manual_cntl(self):
        '''
        declare model control in SymPy Notation if not using dynamics library
        '''

        # Example placeholder is for a system with one control variable
        u1 = sp.symbols('u1')

        # Declare the controls in aggregated vector form:
        u = sp.Matrix([u1])
        shape_u = u.shape # Get the shape of x and return it

        # Declare your controls with given names
        # cntl_dict = {u1:'torque', 'dumb_key':'torque2'}
        cntl_dict = {u1:'torque'}

        # return the controls in aggregation and the shape of the control vector:
        return [u, shape_u, cntl_dict]

    def _declare_manual_params(self):
        '''
        declare all model parameters in SymPy Notation if not using dynamics library
        '''

        # Example pacehold is for a system with three parameters
        m, l, g = sp.symbols(['m', 'l', 'g'])

        # Declare the parameters in aggregate vector form
        params = sp.Matrix([m, l, g])
        shape_params = params.shape # Get the parameter vector dimension

        # Declare your parameters with given names:
        param_dict = {m:'mass', l: 'string length', g: 'acceleration due to gravity'}

        # return the parameters in aggregation and the shape.
        return [params, shape_params, param_dict]


    def _declare_manual_func_f(self):
        '''
        declare all the model dynamics function regularizaiton information using the SymPy notation
        '''

        # This is expecting a user-defined input, so if you have a different model, you will need to change this
        f_expr = sp.Matrix([[self.x[1]], [-1*sp.sin(self.x[0])*self.params[2]/self.params[1]]])

        return f_expr

    def _declare_manual_func_g(self):
        '''
        declare all the model dynamics function control weight information using the SymPy notation
        '''

        # This is expecting a user-defined input, so if you have a different model, you will need to change this
        g_expr = sp.Matrix([[0], [1/(self.params[0]*self.params[1]**2)]])

        return g_expr

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
        f_expr = func_dict["f_expr"] # Find "f_expr" function value
        return f_expr

    def _declare_func_g(self, func_dict):
        '''
        declare all the model dynamics in the specificiation
        '''
        g_expr = func_dict["g_expr"]

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

    def linearize_dynamics(self, **kwargs):
        '''
        linearize the dynamics of a nonlinear continuous model with respect to the state and control
        '''
        # Pull the substitution arguments, if provided, else assume linearization around zero state
        x_sub = kwargs.get('x_sub', sp.zeros(self.shape_x[0], self.shape_x[1]))
        # Pull the substitution arguments, if provided, else assume linearization around zero control
        u_sub = kwargs.get('u_sub', sp.zeros(self.shape_u[0], self.shape_u[1]))

        x_sub = sp.Matrix(x_sub) # Casting this to a SymPy List for subtraction later
        u_sub = sp.Matrix(u_sub) # Casting this to a SymPy List for subtraction later

        if self.disc_flag:
            jacobian_x = super()._take_jacobian(self.disc_model, self.x) # Take the Jacobian wrt the states
            jacobian_u = super()._take_jacobian(self.disc_model, self.u) # Take the Jacobian wrt the control
            evaled = self._sub_vals(self.disc_model, self.x, x_sub) # sub state to generate Steady State Value
            evaled = self._sub_vals(evaled, self.u, u_sub) # sub control to generate Steady State Value
        else:
            jacobian_x = super()._take_jacobian(self.cont_model, self.x) # Take the Jacobian wrt the states
            jacobian_u = super()._take_jacobian(self.cont_model, self.u) # Take the Jacobian wrt the control
            evaled = self._sub_vals(self.cont_model, self.x, x_sub) # sub state to generate Steady State Value
            evaled = self._sub_vals(evaled, self.u, u_sub) # sub control to generate Steady State Value            

        self.jacobian_x = jacobian_x # Store the Jacobian with respect to x
        self.jacobian_u = jacobian_u # Store the Jacobian with respect to u

        # Symbolic Jacobian's for us in symbolic manipulation
        jacobian_x = self._sub_vals(jacobian_x, self.x, x_sub)
        jacobian_u = self._sub_vals(jacobian_u, self.u, u_sub)
        evaled = self._sub_vals(evaled, self.x, self.u)

        # Evaluate the symbolic Jacobian at the linearization point
        jacobian_x = jacobian_x.subs(self.x[0], x_sub[0])
        jacobian_u = jacobian_u.subs(self.u[0], u_sub[0])

        ss_val = evaled # This is the steady-state component

        # Provide symbolic linear model
        self.lin_model = ss_val + jacobian_x*(self.x - x_sub) + jacobian_u*(self.u - u_sub)
        self.lin_measure_model = x_sub + sp.eye(self.shape_x[0])*(self.x-x_sub)
        self.lin_model_lam = super()._convert_funcs2lam(self.lin_model)
        self.lin_measure_model_lam = super()._convert_funcs2lam(self.lin_measure_model)

        # Generate lambda function for Jacobian of x
        self.jac_x_lam = super()._convert_funcs2lam(self.jacobian_x)
        # Generate lambda function for Jacobian of u
        self.jac_u_lam = super()._convert_funcs2lam(self.jacobian_u)