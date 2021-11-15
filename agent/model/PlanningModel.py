from abc import abstractmethod
import numpy as np
from abc import ABC, abstractmethod

class PlanningModel(ABC):
    '''
        xt+1 = A(xt)xt + B(xt)ut
    '''
    def __init__(self, spec: dict) -> None:
        self.spec = spec
        self._state_component = spec['state_component']

    @property
    def state_component(self):
        return self._state_component

    @abstractmethod
    def A(self, x: np.array) -> np.array:
        pass

    @abstractmethod
    def B(self, x: np.array) -> np.array:
        pass

class DoubleIntegrator(PlanningModel):
    '''
        Defines a double integrator for 1D state
    '''
    def __init__(self, spec: dict) -> None:
        super().__init__(spec)

        self.dt = self.spec['dT']
        self.udim = 1
    
    def set_dt(self, dt: float) -> None:
        self.dt = dt

    def A(self, x: np.array = None, dt: float = None) -> np.array:

        if dt is None:
            dt = self.dt

        return np.asarray(
            [[1, dt],
             [0, 1 ]]
        )

    def B(self, x: np.array = None, dt: float = None) -> np.array:
        
        if dt is None:
            dt = self.dt

        return np.asarray(
            [[ 0.5*( dt**2 ) ],
             [ dt]]
        )
