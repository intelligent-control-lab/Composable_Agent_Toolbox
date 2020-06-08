import numpy as np
#Chase
class Model(object):
    def __init__(self, spec):
        self.x_shape = (4,1)
        self.u_shape = (2,1)

    def _f(self, x):
        return np.vstack([x[2], x[3], 0, 0])

    def _g(self, x):
        B = np.matrix(np.zeros((4,2)))
        B[2,0] = 1
        B[3,1] = 1
        return B

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
        