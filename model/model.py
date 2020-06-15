import numpy as np
import sympy as sp

class model(object):
    def __init__(self, use_lib, mod_name, nonlin):
        '''
        class constructor for model class.

        This class constructor looks for two arguments:
            use_lib - exp: boolean - whether to leverage dynamics library
            mod_name - exp: string - name of the model (in dynamics library)
        '''

        # Flag for denoting hand-defined dynamics
        self.use_lib = use_lib
        self.mod_name = mod_name
        self.nonlin   = nonlin
        if self.use_lib:
            # Perform a look-up in the dynamics library 
            pass
        else: 
            state_dec = self._declareState()
            self.x = state_dec[0] # This defines the state vector
            self.shape_x = state_dec[1] # This is the shape of the state vector

            cntl_dec = self._declareCntl()
            self.u = cntl_dec[0] # This defines the state vector
            self.shape_u = cntl_dec[1] # This is the shape of the control vector
            params_dec = self._declareParams()
            self.params = params_dec[0]
            self.shape_params = params_dec[1]

            # Determine if this a nonlinear system or a linear system
            if self.nonlin:
                # Declare f(x, params)
                self.f_expr = self._declareFuncf(self.x, self.params)
                self.g_expr = self._declareFuncg(self.x, self.params)
                self.cont_model = self.f_expr + self.g_expr*self.u 
            else: 
                # This is a linear system, will implement this later
                pass


    def _declareState(self):
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

    def _declareCntl(self):
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

    def _declareParams(self):
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


    def _declareFuncf(self, x, params):
        '''
        declare all the model dynamics function regularizaiton information using the SymPy notation
        '''

        f_expr = sp.Matrix([[x[0]], [-1*sp.sin(x[0])*params[2]/params[1]]])
 
        return f_expr

    def _declareFuncg(self, x, params):
        '''
        declare all the model dynamics function control weight information using the SymPy notation
        '''

        g_expr = sp.Matrix([[0], [1/(params[0]*params[1]**2)]])

        return g_expr

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
    pen_model = model(0, 'nonlin_pen', 1)
    print(pen_model.mod_name)
    print(pen_model.u)
    print(pen_model.cont_model)