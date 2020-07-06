class Controller_Manager(object):
    '''
    Define the structure of the controller.
    The user could modify the `build_controller` as needed.
    '''
    def __init__(self, coordination, feedforward, feedback, safety):
        self.coordination = coordination
        self.feedforward = feedforward
        self.feedback = feedback
        self.safety = safety

    def build_controller(self, dt, x, goal_x, est_params):
        '''
        Modify this function according to the controller structure.
        
        E.g. a feedback only controller
        feedback_output = self.feedback.control(dt, x[0], goal_x[0], est_params)
        return feedback_output
        '''
        # coordination_output = self.coordination.control(dt, x, goal_x, est_params)
        # feedforward_output  = self.feedforward.control(dt, coordination_output['x'], coordination_output['goal_x'], est_params)
        # feedback_output     = self.feedback.control(dt, x, goal_x, est_params)
        # safety_output       = self.safety.control(dt, x[0], goal_x[0], est_params)
        feedback_output = self.feedback.control(dt, x, goal_x, est_params)
        return feedback_output