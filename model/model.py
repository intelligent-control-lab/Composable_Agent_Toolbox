# Declare imports
# Style guide is currently following PEP8 - which means sort alphabetically

# Standard Library Import
from abc import ABC, abstractmethod  # This is needed for the abstract classes

# Third-party Imports
import numpy as np
import sympy as sp

# Application Specific Imports
# None

class ModelBase(ABC):
    ''' 
    define an base class for developing models

    Classes derived from the model base need to implement copies
    of all abstract methods, lest a TypeError arises.
    '''

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _declare_state(self):
        pass

    @abstractmethod
    def _declare_cntl(self):
        pass

    @abstractmethod
    def _declare_params(self):
        pass

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
        lam_func = sp.lambdify(self.all_syms,func)
        return lam_func

    def _take_jacobian(self, func, VARS):
        '''
        this method linearizes the system dynamics by taking the jacobian of func with respect to argument X and returns a symbolic object
        '''

        # Leverages the SymPy dynamics structure
        return func.jacobian(VARS)

    def evaluate_dynamics(self, x_sub, u_sub, params_sub):
        '''
        evaluate the dynamics with a specific values substituted for the state and the parameters

        Note that this is a particularly slow way of evaluating the dynamics, but it does work if you want the symbolic objects
        '''
        if self.disc_flag:
            evaled = self.disc_model.subs(self.x[0], x_sub[0]) # Do a first element substitution so we can get an object to iterate over
            counter = 0 # This is a counter to iterate through the elements in the state vector
            for i in self.x:
                evaled = evaled.subs(i, x_sub[counter]) # Sub in each corresponding element in the state vector with that in passed x_sub arg
                counter += 1 # increment counter
            counter = 0 # reset counter for use in the controls substitution
            for i in self.u:
                evaled = evaled.subs(i, u_sub[counter]) # Sub in each corresponding element in the control vector with that in passed u_sub arg
                counter += 1 # increment counter
            counter = 0 # reset counter for use in the parameters substitution
            for i in self.params:
                evaled = evaled.subs(i, params_sub[counter]) # Sub in each corresponding element in the parameter vector with that in the passed parameter arg
                counter += 1                
            return [evaled, np.array(evaled).astype(np.float64)] # return a symbolic math object (for manipulation) as well as a casted numpy object
        else: 
            # Evaluate the expression using a continuous time flag
            evaled = self.cont_model.subs(self.x, x_sub)
            evaled = evaled.subs(self.u, u_sub)            
            evaled = evaled.subs(self.params, params_sub)
            return [evaled, np.array(evaled).astype(np.float64)]

    def evaluate_measurement(self, x_sub):
        '''
        evaluate the measurement function with a specific value substituted fro the state and parameters. Currently assumes full-state feedback
        '''
        evaled = self.measure_func.subs(self.x[0], x_sub[0]) # Do a first element substitution so we can get an object to iterate over
        counter = 0 # This is a counter to iterate through the elements in the state vector
        for i in self.x:
            evaled = evaled.subs(i, x_sub[counter]) # sub in each corresponding element in the state vector with that in passed x_sub arg
            counter += 1
        return [evaled, np.array(evaled).astype(np.float64)] # return a symbolic math object (for symbolic manipulation) as well as a casted numpy object    


class NonlinModelCntlAffine(ModelBase):
    '''
    class for nonlinear control affine models of form x_dot = f(x) + g(x)u
    '''
    def __init__(self, use_library, model_name, time_sample, disc_flag):
        '''
        class constructor for model class.

        This class constructor looks for two arguments:
            use_library - exp: boolean - whether to leverage dynamics library
            model_name - exp: string - name of the model (in dynamics library)
            time_sample - exp: double/float - time sample value
            disc_flag - exp: boolean - discrete time flag modifier
        '''

        # Flag for denoting hand-defined dynamics
        self.use_library = use_library
        self.model_name = model_name
        self.time_sample = time_sample
        self.disc_flag = disc_flag
        self.symbol_dict = {} # Define an empty dictionary for holding the variables
        self.lin_model = None # Only populate if we actually linearize
        self.lin_measure_model = None # Only populate if we actually linearize
        self.lin_model_lam = None # Only populate if we actually linearize
        self.lin_measure_model_lam = None # Only populate if we actually linearize        
        self.jac_x_lam = None # Only populate if we actually linearize
        self.jac_u_lam = None # Only populate if we actually linearize

        if self.use_library:
            # Perform a look-up in the dynamics library 
            pass
        else:
            # Initialize States
            state_dec = self._declare_state() # Function initializes state variables
            self.x = state_dec[0] # This defines the state vector and gives it to the model object
            self.shape_x = state_dec[1] # This is the shape of the state vector
            self.symbol_dict.update(state_dec[2]) # This updates the symbol dictionary to include the state symbol to mapping information

            # Initialize Controls
            cntl_dec = self._declare_cntl()
            self.u = cntl_dec[0] # This defines the control vector
            self.shape_u = cntl_dec[1] # This is the shape of the control vector
            self.symbol_dict.update(cntl_dec[2]) # This updates the symbol dictionary to include the control symbol to name mapping information

            # Initialize Parameters
            params_dec = self._declare_params()
            self.params = params_dec[0] # This defines the parameter vector
            self.shape_params = params_dec[1] # This is the shape of the parameter vector
            self.symbol_dict.update(params_dec[2]) # This updates the symbol dictionary to include the parameter symbol to name mapping information

            # Generate a list of all in use symbols
            self.all_syms = [[self.x, self.u, self.params]] # "Vertically stack" the states, parameter, control (really a three element list)

            # Determine if this a nonlinear system or a linear system
            # Currently, this assumes a system of the form: xdot = f(x) + g(x)u
            self.f_expr = self._declare_func_f() # Defines f(x) as it would appear in xdot = f(x) + g(x)u
            self.g_expr = self._declare_func_g() # Defines g(x) as it would appear in xdot = f(x) + g(x)u
            self.cont_model = self.f_expr + self.g_expr*self.u # Actually form the Right Hand Side (RHS) xdot = f(x) + g(x)u
            self.measure_func = self._declare_func_measure() # Defines y = C*x - Currently only supports Full State Feedback

            if self.disc_flag:  # Discretize the System
                self.disc_model = super()._discretize_dyn(self.cont_model) # Discretizes the dynamics with a forward-Euler approximation
                self.cont_model_lam = super()._convert_funcs2lam(self.cont_model) # Converts the dynamics equations into a Python Lambda function (anonymous function)
                self.disc_model_lam = super()._convert_funcs2lam(self.disc_model) # Converts the dynamics equations into a Python Lambda function (anonymous function)
                self.measure_func_lam = super()._convert_funcs2lam(self.measure_func) # Converts the measurement equations into a Python Lambda function (anonymous function)
            else: 
                # Evaluate the system in a continuous form:
                pass

    def _declare_state(self):
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

    def _declare_cntl(self):
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

    def _declare_params(self):
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


    def _declare_func_f(self):
        '''
        declare all the model dynamics function regularizaiton information using the SymPy notation
        '''

        # This is expecting a user-defined input, so if you have a different model, you will need to change this
        f_expr = sp.Matrix([[self.x[1]], [-1*sp.sin(self.x[0])*self.params[2]/self.params[1]]])
 
        return f_expr

    def _declare_func_g(self):
        '''
        declare all the model dynamics function control weight information using the SymPy notation
        '''

        # This is expecting a user-defined input, so if you have a different model, you will need to change this
        g_expr = sp.Matrix([[0], [1/(self.params[0]*self.params[1]**2)]])

        return g_expr

    def _declare_func_measure(self):
        '''
        declare all measurement outputs for the system
        '''

        # Currently, this returns a full-state feedback system, future implementations will likely require different measurement functions
        return sp.eye(self.shape_x[0])*self.x

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
        x_sub = kwargs.get('x_sub', sp.zeros(self.shape_x[0], self.shape_x[1])) # Pull the substitution arguments, if provided, else assume linearization around zero state
        u_sub = kwargs.get('u_sub', sp.zeros(self.shape_u[0], self.shape_u[1])) # Pull the substitution arguments, if provided, else assume linearization around zero control

        x_sub = sp.Matrix(x_sub) # Casting this to a SymPy List for subtraction later
        u_sub = sp.Matrix(u_sub) # Casting this to a SymPy List for subtraction later

        if self.disc_flag:
            jacobian_x = super()._take_jacobian(self.disc_model, self.x) # Take the Jacobian wrt the states
            jacobian_u = super()._take_jacobian(self.disc_model, self.u) # Take the Jacobian wrt the control
            evaled = self.disc_model.subs(self.x[0], x_sub[0]) # Do a first element substitution so we can get an object to iterate over
        else:
            jacobian_x = super()._take_jacobian(self.cont_model, self.x) # Take the Jacobian wrt the states
            jacobian_u = super()._take_jacobian(self.cont_model, self.u) # Take the Jacobian wrt the control
            evaled = self.cont_model.subs(self.x[0], x_sub[0]) # Do a first element substitution so we can get an object to iterate over

        self.jacobian_x = jacobian_x # Store the Jacobian with respect to x
        self.jacobina_u = jacobian_u # Store the Jacobian with respect to u

        jacobian_x = jacobian_x.subs(self.x[0], x_sub[0])
        jacobian_u = jacobian_u.subs(self.u[0], u_sub[0])
        counter = 0 # This is a counter to iterate through the elements in the state vector
        for i in self.x:
            evaled = evaled.subs(i, x_sub[counter]) # Sub in each corresponding element in the state vector with that in passed x_sub arg
            jacobian_x = jacobian_x.subs(i, x_sub[counter])
            jacobian_u = jacobian_u.subs(i, x_sub[counter])
            counter += 1 # increment counter
        counter = 0 # reset counter for use in the controls substitution
        for i in self.u:
            evaled = evaled.subs(i, u_sub[counter]) # Sub in each corresponding element in the control vector with that in passed u_sub arg
            jacobian_x = jacobian_x.subs(i, u_sub[counter])
            jacobian_u = jacobian_u.subs(i, u_sub[counter])
            counter += 1 # increment counter
 
        ss_val = evaled # This is the steady-state component

        # Provide symbolic linear model
        self.lin_model = ss_val + jacobian_x*(self.x - x_sub) + jacobian_u*(self.u - u_sub)
        self.lin_measure_model = x_sub + sp.eye(self.shape_x[0])*(self.x-x_sub)
        self.lin_model_lam = super()._convert_funcs2lam(self.lin_model)
        self.lin_measure_model_lam = super()._convert_funcs2lam(self.lin_measure_model)

        self.jac_x_lam = super()._convert_funcs2lam(self.jacobian_x) # This is a lambda function for Jacobian of x
        self.jac_u_lam = super()._convert_funcs2lam(self.jacobian_u) # This is a lambda function for Jacobian of u
 
class LinearModel(ModelBase):
    '''
    class for Linear Models of form x_dot = Ax + Bu
    '''
    def __init__(self, use_library, model_name, time_sample, disc_flag):
        '''
        class constructor for model class.

        This class constructor looks for two arguments:
            use_library - exp: boolean - whether to leverage dynamics library
            model_name - exp: string - name of the model (in dynamics library)
            time_sample - exp: double/float - time sample value
            disc_flag - exp: boolean - discrete time flag modifier
        '''

        # Flag for denoting hand-defined dynamics
        self.use_library = use_library
        self.model_name = model_name
        self.time_sample = time_sample
        self.disc_flag = disc_flag
        self.symbol_dict = {} # Define an empty dictionary for holding the variables

        if self.use_library:
            # Perform a look-up in the dynamics library 
            pass
        else:
            # Initialize States
            state_dec = self._declare_state() # Function initializes state variables
            self.x = state_dec[0] # This defines the state vector and gives it to the model object
            self.shape_x = state_dec[1] # This is the shape of the state vector
            self.symbol_dict.update(state_dec[2]) # This updates the symbol dictionary to include the state symbol to mapping information

            # Initialize Controls
            cntl_dec = self._declare_cntl()
            self.u = cntl_dec[0] # This defines the control vector
            self.shape_u = cntl_dec[1] # This is the shape of the control vector
            self.symbol_dict.update(cntl_dec[2]) # This updates the symbol dictionary to include the control symbol to name mapping information

            # Initialize Parameters
            params_dec = self._declare_params()
            self.params = params_dec[0] # This defines the parameter vector
            self.shape_params = params_dec[1] # This is the shape of the parameter vector
            self.symbol_dict.update(params_dec[2]) # This updates the symbol dictionary to include the parameter symbol to name mapping information

            # Generate a list of all in use symbols
            self.all_syms = [[self.x, self.u, self.params]] # "Vertically stack" the states, parameter, control (really a three element list)

            # Determine if this a nonlinear system or a linear system
            # Currently, this assumes a system of the form: xdot = f(x) + g(x)u
            self.A = self._declare_func_A() # Defines f(x) as it would appear in xdot = f(x) + g(x)u
            self.B = self._declare_func_B() # Defines g(x) as it would appear in xdot = f(x) + g(x)u
            self.cont_model = self.A*self.x + self.B*self.u # Actually form the Right Hand Side (RHS) xdot = f(x) + g(x)u
            self.measure_func = self._declare_func_measure() # Defines y = C*x - Currently only supports Full State Feedback

            if self.disc_flag:  # Discretize the System
                self.disc_model = super()._discretize_dyn(self.cont_model) # Discretizes the dynamics with a forward-Euler approximation
                self.cont_model_lam = super()._convert_funcs2lam(self.cont_model) # Converts the dynamics equations into a Python Lambda function (anonymous function)
                self.disc_model_lam = super()._convert_funcs2lam(self.disc_model) # Converts the dynamics equations into a Python Lambda function (anonymous function)
                self.measure_func_lam = super()._convert_funcs2lam(self.measure_func) # Converts the measurement equations into a Python Lambda function (anonymous function)
            else: 
                # Evaluate the system in a continuous form:
                pass

    def _declare_state(self):
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

    def _declare_cntl(self):
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

    def _declare_params(self):
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

    def _declare_func_A(self):
        '''
        declare all the model dynamics function regularizaiton information using the SymPy notation
        '''

        # This is expecting a user-defined input, so if you have a different model, you will need to change this
        A = sp.Matrix([[0,1,0,0], [0,0,0,0], [0,0,0,1],[0,0,0,0]])
 
        return A

    def _declare_func_B(self):
        '''
        declare all the model dynamics function control weight information using the SymPy notation
        '''

        # This is expecting a user-defined input, so if you have a different model, you will need to change this
        B = sp.Matrix([[0,0], [1/self.params[0], 0], [0,0],[0,1/self.params[0]]])

        return B

    def _declare_func_measure(self):
        '''
        declare all measurement outputs for the system
        '''

        # Currently, this returns a full-state feedback system, future implementations will likely require different measurement functions
        return sp.eye(self.shape_x[0])*self.x

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