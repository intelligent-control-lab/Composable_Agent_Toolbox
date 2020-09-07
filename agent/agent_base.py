from abc import ABC, abstractmethod

# Agent Base Class
class AgentBase(ABC):
  
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def action(self, dt, sensors_data):
        pass