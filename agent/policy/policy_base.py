# define the actor network
from abc import ABC, abstractmethod

class PolicyBase(ABC):
    @abstractmethod
    def __init__(self, policy_spec):
        pass
    
    @abstractmethod
    def action(self, dt, sensor_data):
        pass
    