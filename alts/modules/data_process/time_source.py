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
    
    def __post_init__(self):
        self._time_offset: float = self.start_time
        self._iter = 0

    def step(self, iteration: int) -> NDArray[Shape["time_step_nr, [time]"], Number]:
        self._iter = iteration
        return np.asarray([[self.time]])
    
    @property
    def time(self)-> float:
        return self._iter*self.time_step + self._time_offset
    
    @time.setter
    def time(self, time):
        self._time_offset = self.time - time