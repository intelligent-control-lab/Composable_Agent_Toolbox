import numpy as np

class PVSensor(object):
    def __init__(self, target, spec):
        self.target = target
    def _noise(self, x):
        return x + (np.random.random_sample(x.shape) * 0.05 - 0.05)
    def measure(self):
        pos = self.target.pos
        vel = self.target.vel
        return self._noise(pos), self._noise(vel)
