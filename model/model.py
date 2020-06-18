
# Declare imports
import numpy as np
import sympy as sp
import math as m

class Model(object):
    def __init__(self, use_library, model_name, nonlinear_flag, time_sample, disc_flag):
        '''
        class constructor for model class.

        This class constructor looks for two arguments:
            use_library - exp: boolean - whether to leverage dynamics library
            model_name - exp: string - name of the model (in dynamics library)
            nonlinear_flag - exp: boolean - whether or not to use nonlinear dynamic structure
            time_sample - exp: double/float - time sample value
            disc_flag - exp: boolean - discrete time flag modifier
        '''

        # Flag for denoting hand-defined dynamics
        self.use_library = use_library
        self.model_name = model_name
        self.nonlinear_flag   = nonlinear_flag
        self.time_sample = time_sample
        self.disc_flag = disc_flag

        # Declare the 
        if self.use_library:
            # Perform a look-up in the dynamics library 
            pass
        else: 
            state_dec = self._declare_state()
            self.x = state_dec[0] # This defines the state vector
            self.shape_x = state_dec[1] # This is the shape of the state vector

            cntl_dec = self._declare_cntl()
            self.u = cntl_dec[0] # This defines the state vector
            self.shape_u = cntl_dec[1] # This is the shape of the control vector
            params_dec = self._declare_params()
            self.params = params_dec[0]
            self.shape_params = params_dec[1]

            # Determine if this a nonlinear system or a linear system
            if self.nonlinear_flag:
                # Declare f(x, params)
                self.f_expr = self._declare_func_f()
                self.g_expr = self._declare_func_g()
                self.cont_model = self.f_expr + self.g_expr*self.u 
                self.measure_func = self._declare_func_measure()
                if self.disc_flag: 
                    self.disc_model = self._discretize_dyn()
                else: 
                    # Evaluate the system in a continuous form:
                    pass
            else:
                # This is a linear system, will implement this later
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

        # return the states in aggregation and the shape of the state vector:
        return [x, shape_x]

    def _declare_cntl(self):
        '''
        declare model control in SymPy Notation if not using dynamics library
        '''

        # Example placeholder is for a system with one control variable
        u1 = sp.symbols(['u1'])

        # Declare the controls in aggregated vector form:
        u = sp.Matrix([u1])
        shape_u = u.shape # Get the shape of x and return it

        # return the controls in aggregation and the shape of the control vector:
        return [u, shape_u]

    def _declare_params(self):
        '''
        declare all model parameters in SymPy Notation if not using dynamics library
        '''

        # Example pacehold is for a system with three parameters
        m, l, g = sp.symbols(['m', 'l', 'g'])

        # Declare the parameters in aggregate vector form
        params = sp.Matrix([m, l, g])
        shape_params = params.shape # Get the parameter vector dimension

        # return the parametesr in aggregation and the shape.
        return [params, shape_params]


    def _declare_func_f(self):
        '''
        declare all the model dynamics function regularizaiton information using the SymPy notation
        '''

        f_expr = sp.Matrix([[self.x[1]], [-1*sp.sin(self.x[0])*self.params[2]/self.params[1]]])
 
        return f_expr

    def _declare_func_g(self):
        '''
        declare all the model dynamics function control weight information using the SymPy notation
        '''

        g_expr = sp.Matrix([[0], [1/(self.params[0]*self.params[1]**2)]])

        return g_expr

    def _declare_func_measure(self):
        '''
        declare all measurement outputs for the system
        '''
        return sp.eye(self.shape_x[0])*self.x

    def _discretize_dyn(self):
        '''
        use a first order Euler Approximation to discretize the system dynamics
        '''
        dis_model = self.x + self.time_sample*self.cont_model
        return dis_model

    def evaluate_dynamics(self, x_sub, u_sub, params_sub):
        '''
        evaluate the dynamics with a specific values substituted for the state and the parameters
        '''
        if self.disc_flag:
            evaled = self.disc_model.subs(self.x[0], x_sub[0])
            counter = 0
            for i in self.x:
                evaled = evaled.subs(i, x_sub[counter])
                counter += 1
            counter = 0
            for i in self.u:
                evaled = evaled.subs(i, u_sub[counter])
                counter += 1
            counter = 0
            for i in self.params:
                evaled = evaled.subs(i, params_sub[counter])
                counter += 1                
            return [evaled, np.array(evaled).astype(np.float64)]
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
        evaled = self.measure_func.subs(self.x[0], x_sub[0])
        counter = 0
        for i in self.x:
            evaled = evaled.subs(i, x_sub[counter])
            counter += 1
        return [evaled, np.array(evaled).astype(np.float64)] 

    def forward(self, x):
        '''
        dot_x = f(x) + g(x) u
        '''
        return self._f(x), self._g(x)

    def inverse(self, p, x):
        '''
        dp_dx: derivative of the robot's cartesian state to its internal state,
               in the dodge obstacle task, the cartesian state is set as the
               closed point on the robot to the obstacle.
        '''
        dp_dx = np.zeros((2,4))
        return dp_dx

if __name__ == '__main__':
    print("This is the main, do not wear it out!")
    pen_model = Model(0, 'nonlin_pen', 1, 0.01, 1)
    print(pen_model.model_name)
    print(pen_model.u)
    print(pen_model.cont_model)
    print(pen_model.disc_model)
    print(pen_model.measure_func)
    evaled = pen_model.evaluate_dynamics([m.pi, 1], [0.01], [2, 9.81, 9.81])
    print('Value of the state when placed in the dynamics')
    print(evaled)
    print('Value of the expected measurement equation')
    evaled_measure = pen_model.evaluate_measurement([m.pi, 1])
    print(evaled_measure)