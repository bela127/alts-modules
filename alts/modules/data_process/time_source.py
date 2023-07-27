from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass,field

import numpy as np

from alts.core.data_process.time_source import TimeSource
from alts.core.configuration import ConfAttr, pre_init

if TYPE_CHECKING:
    from nptyping import NDArray, Shape, Number

@dataclass
class IterationTimeSource(TimeSource):

    start_time: float = 0
    time_step: float = 1
    
    def post_init(self):
        super().post_init()
        self._time_offset: float = self.start_time
        self._iter = 0

    def step(self, iteration: int) -> NDArray[Shape["time_step_nr, [time]"], Number]:
        super().step(iteration = iteration)
        self._iter = iteration
        return np.asarray([[self.time]])
    
    @property
    def time(self)-> float:
        return self._iter*self.time_step + self._time_offset
    
    @time.setter
    def time(self, delta_time):
        if delta_time > 0:
            self._time_offset += delta_time
        else:
            raise ValueError("Time can only be incremented, time travel into the past is not possible!")