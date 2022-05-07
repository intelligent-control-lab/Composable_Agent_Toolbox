import time
import progressbar

class MPWrapper(object):
    
    def __init__(self, env):
        self.env = env

    def reset(self):
        return self.env.reset()

    def step_loop(self, mgr_actions, mgr_sensor_data, mgr_record, mgr_running, lock, iters, render=True):
        
        print("Simulation progress:")
        for it in progressbar.progressbar(range(iters)):
            actions = {}
            with lock:
                actions.update(mgr_actions)
            # control = actions['robot']['control']
            # print(f'env wrapper:\n{control}')
            dt, env_info, sensor_data = self.env.step(actions, render=render)
            sensor_data['time'] = time.time()
            mgr_record.put((env_info, sensor_data))
            with lock:
                mgr_sensor_data.update(sensor_data)

        with lock:
            mgr_running.value = False
