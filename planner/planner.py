#Weiye
class Planner(object):
    def __init__(self, spec, model):
        self.spec = spec
        self.model = model
        self.cache = {}
    def planning(self, dt, goal, agent_state):
        agent_next_state = [agent_state] * 100
        return agent_next_state
    def re_planning(self, dt, goal, agent_state):
        agent_next_state = [agent_state] * 100
        return agent_next_state
