import time

class MPWrapper(object):
    
    def __init__(self, agent):
        self.agent = agent
        self.last_time = time.time()
        self.last_cycle = 0
        self.cycle_time = agent.module_spec["cycle_time"] # maybe init somewhere else?

    def init_action(self, sensor_data):
        self.last_time = time.time()
        dt = 0
        action = self.agent.action(dt, sensor_data)
        action['dt'] = dt
        return action

    def action_loop(self, mgr_actions, mgr_sensor_data, mgr_running, lock):
        
        i = 0
        while True:
            with lock: 
                if not mgr_running.value: 
                    break

            dt = time.time() - self.last_time
            self.last_time += dt
            self.last_cycle += dt
            if self.last_cycle < self.cycle_time:
                time.sleep(0.001)
                continue
            i += 1
            self.last_cycle = 0

            sensor_data = {}
            with lock:
                sensor_data.update(mgr_sensor_data[self.agent.name])
            actions = self.agent.action(dt, sensor_data)
            actions['dt'] = dt
            with lock:
                mgr_actions.update(actions)

