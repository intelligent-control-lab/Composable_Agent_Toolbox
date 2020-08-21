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
            # Evaluate the system in a continuous form:
            pass

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
                # Evaluate the system in a continuous form:
                pass

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
            # Evaluate the system in a continuous form:
            pass

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