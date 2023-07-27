from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

import numpy as np

import GPy

from alts.core.oracle.data_behavior import DataBehavior

if TYPE_CHECKING:
    from typing import Tuple
    from nptyping import  NDArray, Number, Shape


@dataclass
class EquidistantTimeUniformBehavior(DataBehavior):

    def behavior(self) -> Tuple[NDArray[Shape["change_times"], Number], NDArray[Shape["kps"], Number]]:
        numberOfLoadChanges: int = int((self.stop_time-self.start_time) / self.change_interval)
        kp = np.random.uniform(self.lower_value, self.upper_value, numberOfLoadChanges + 1)
        change_time = np.linspace(self.start_time, self.stop_time, numberOfLoadChanges + 1)
        return change_time, kp

    
@dataclass
class RandomTimeUniformBehavior(DataBehavior):

    def behavior(self) -> Tuple[NDArray[Shape["change_times"], Number], NDArray[Shape["var"], Number]]:
        numberOfLoadChanges: int = int((self.stop_time-self.start_time) / self.change_interval)
        var = np.random.uniform(self.lower_value, self.upper_value, (numberOfLoadChanges + 1))
        change_time = np.insert(np.sort(np.random.uniform(self.start_time, self.stop_time, numberOfLoadChanges)), 0, [0])
        return change_time, var


@dataclass
class RandomTimeBrownBehavior(DataBehavior):

    def behavior(self) -> Tuple[NDArray[Shape["change_times"], Number], NDArray[Shape["kps"], Number]]:
        offset = (self.upper_value+self.lower_value)/2
        brown_up = offset
        brown_low = -offset

        numberOfLoadChanges: int = int((self.stop_time-self.start_time) / self.change_interval)

        kp = np.random.uniform(brown_low, brown_up, int(numberOfLoadChanges / 4))
        change_time = np.random.uniform(self.start_time, self.stop_time, int(numberOfLoadChanges / 4))

        k = GPy.kern.Brownian(variance=0.005)
        gp = GPy.models.GPRegression(change_time[:,None], kp[:,None], k, noise_var=0)

        change_time = np.insert(np.sort(np.random.uniform(self.start_time, self.stop_time, numberOfLoadChanges*4)), 0, [0])

        kp = gp.posterior_samples_f(change_time[:,None], size=1)
        offset = (self.upper_value+self.lower_value)/2
        kp = kp[:,0,0]+offset
         
        return change_time, kp

