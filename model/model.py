
#Chase
class Model(object):
    def __init__(self, spec):
        pass
    def forward(self, x):
        '''
        dot_x = f(x) + g(x) u
        '''
        fx = 1
        gx = 1
        return fx, gx

    def inverse(self, p, x):
        '''
        dp_dx: derivative of the robot's cartesian state to its internal state,
               in the dodge obstacle task, the cartesian state is set as the
               closed point on the robot to the obstacle.
        '''
        dp_dx = 1
        return dp_dx
        