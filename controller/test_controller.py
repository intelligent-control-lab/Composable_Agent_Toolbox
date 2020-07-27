import controller
import numpy as np
import control # Python Control System Library

class Model(object):
    def __init__(self, A, B):
        self.A = A
        self.B = B

def test_lqr():
    control_spec = {
                'feedback': 'LQR',
                'params': {
                    'feedback': {
                        'Q': np.array([[1, 0],[0, 1]]),
                        'R': 1
                    }
                }
            }

    model = Model(np.array([[0, 1],[1, 0]]), [[0], [1]])

    dt = 0.01
    x = np.array([[1],[1]])
    goal_x = np.array([[0], [0]])
    est_params = []

    test_controller = controller.Controller(control_spec, model)
    output = test_controller.control(dt, x, goal_x, est_params)


    # expected results
    K, S, E = control.lqr(model.A, model.B, control_spec['params']['feedback']['Q'], control_spec['params']['feedback']['R'])
    expec_output = -K@x

    assert output == expec_output