import time

class MPWrapper(object):
    
    def __init__(self, agent):
        self.agent = agent
        self.name = agent.name
        self.sensors = agent.sensors
        self.last_time = time.time()
        self.last_cycle = 0
        self.cycle_time = agent.module_spec["cycle_time"]

    def init_action(self, sensor_data, debug_modes):
        self.last_time = time.time()
        dt = 0.01
        action = self.agent.action(dt, sensor_data[self.name], debug_modes)
        action['dt'] = dt
        return action

    def action_loop(self, mgr_actions, mgr_sensor_data, mgr_running, lock, debug_modes):
        
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
            dt = self.last_cycle
            self.last_cycle = 0
            i += 1

            sensor_data = {}
            with lock:
                sensor_data = mgr_sensor_data[self.name]
            actions = self.agent.action(dt, sensor_data, debug_modes)
            actions['dt'] = dt
            with lock:
                mgr_actions[self.name] = actions
