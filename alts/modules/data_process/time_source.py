#Version 1.1.1 conform as of 29.11.2024
"""
*alts.modules.data_process.time_Source*
:doc:`Core Module </core/data_process/time_source>`
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass,field

import numpy as np

from alts.core.data_process.time_source import TimeSource
from alts.core.configuration import ConfAttr, pre_init, init

if TYPE_CHECKING:
    from nptyping import NDArray, Shape, Number

@dataclass
class IterationTimeSource(TimeSource):
    """
    IterationTimeSource(start_time, time_step)
    | **Description**
    |   This ``TimeSource`` iterates through time starting at 0 in steps of 1. Time here can be positively offset.
    
    :param start_time: Initial time (default = 0)
    :type start_time: float
    :param time_step: Steps of increment of time (default = 1)
    :type time_step: float
    """
    start_time: float = init(default=0)
    time_step: float = init(default=1)
    
    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Sets all counters to 0.
        """
        super().post_init()
        self._time_offset: float = self.start_time
        self._iter = 0

    def step(self, iteration: int) -> NDArray[Shape["time_step_nr, [time]"], Number]: # type: ignore
        """
        step(self, iteration) -> time
        | **Description**
        |   Sets time to the given ``iteration``.
        |   Example: step(1) -> self.time == 1 + offset, step(3) -> self.time == 3 + offset

        :param iteration: Time will be set to have done this many steps
        :type iteration: int
        :return: New time
        :rtype: float
        """
        super().step(iteration = iteration)
        self._iter = iteration
        return np.asarray([[self.time]])
    
    @property
    def time(self) -> float:
        """
        time(self) -> float
        | **Description**
        |   Returns the current time of the ``TimeSource``
        |   Time is set to be the current iteration times the size of the step each iteration increments by plus a given time offset.
        |   time = iteration * time_step + offset

        :return: Current time
        :rtype: float
        """
        return self._iter*self.time_step + self._time_offset
    
    @time.setter
    def time(self, delta_time):
        """
        time(self, delta_time) -> None
        | **Description**
        |   While time itself is not possible to be set to a certain value at will, this ``TimeSource`` allows a time offset to be set here.
        |   The given positive ``delta_time`` is added to ``time_offset``. 

        :param delta_time: A positive number to offset time by
        :type delta_time: Number
        """
        if delta_time > 0:
            self._time_offset += delta_time
        else:
            raise ValueError("Time can only be incremented, time travel into the past is not possible!")