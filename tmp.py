import numpy as np
from abc import ABC, abstractmethod

class Sensor(ABC):
    @abstractmethod
    def __init__(self, a):
        self.a = a
    
    @abstractmethod
    def do_something(self):
        pass


class FuckSensor(Sensor):
    def __init__(self, a):
        self.b = 1
        super().__init__(a)
    
    def do_something(self):
        print("fuck")
        print(self.a)


f = FuckSensor(5)
f.do_something()