import numpy as np

class Evaluator(object):
    def __init__(self, agent_specs, env_spec):
        self.agent_specs = agent_specs
        self.env_spec = env_spec

    def evaluate(self, record):
        
        achieved_goal = record[-1][0]["robot_goal"]["count"]        
        safety = 0
        d0 = 1
        
        for i in range(len(record)):
            min_dis = 1e9
            min_name = ""
            for name, pos_vel in record[i][1]["robot"]["obstacle_sensor"].items():
                dis = pos_vel["rel_pos"]
                if np.linalg.norm(dis) < min_dis:
                    min_dis = np.linalg.norm(dis)
                    min_name = name
            safety += min(0, np.log(min_dis / d0))

        print("===================================")
        print("achieved goal:", achieved_goal)
        print("safety score:", safety)
        print("===================================")
        return 0
