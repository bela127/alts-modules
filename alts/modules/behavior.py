#Version 1.1.1 conform as of 29.11.2024
"""
*alts.modules.behavior*
:doc:`Core Module </core/oracle/data_behaviour>`
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

import numpy as np

import GPy

from alts.core.oracle.data_behavior import DataBehavior

if TYPE_CHECKING:
    from typing import Tuple
    from nptyping import  NDArray, Shape


@dataclass
class EquidistantTimeUniformBehavior(DataBehavior):
    """
    EquidistantTimeUniformBehavior(change_interval, lower_value, upper_value, start_time, stop_time)
    | **Description**
    |   Changes data behaviour in regular intervals by a uniformly random value between lower_value and upper_value

    :param change_interval: Amount of time steps between behaviour changes (default=5.0)
    :type change_interval: float
    :param lower_value: Lower extent of data value changes, inclusive (default=-1.0)
    :type lower_value: float
    :param upper_value: Upper extent of data value changes, exclusive (default=1.0)
    :type upper_value: float
    :param start_time: Start of affected time (default=0.0)
    :type start_time: float
    :param stop_time: Stop of affected time (default=600.0)
    :type stop_time: float
    """
    def behavior(self) -> Tuple[NDArray[Shape["change_times"], np.dtype[np.number]], NDArray[Shape["kps"], np.dtype[np.number]]]:
        """
        behavior(self) -> change_times, change_values
        | **Description**
        |   First calculates how many times the data changes ``=n``.
        |   Then draws ``n+1``[#]_ many uniformly random values in [lower_value, upper_value) ``=change_values``.
        |   Finally calculates the equidistant change times based on ``n+1`` ``=change_times``

        :return: change_times, change_values
        :rtype: `NDArray[float] <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_, `NDArray[float] <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        
        .. [#] It is n+1 as the first behaviour change sets in at start_time
        """
        numberOfLoadChanges: int = int((self.stop_time-self.start_time) / self.change_interval)
        kp = np.random.uniform(self.lower_value, self.upper_value, numberOfLoadChanges + 1)
        change_time = np.linspace(self.start_time, self.stop_time, numberOfLoadChanges + 1)
        return change_time, kp

    
@dataclass
class RandomTimeUniformBehavior(DataBehavior):
    """
    RandomTimeUniformBehavior(change_interval, lower_value, upper_value, start_time, stop_time)
    | **Description**
    |   Changes data behaviour in random intervals by a uniformly random value between lower_value and upper_value

    :param change_interval: Average amount of time steps between behaviour changes (default=5.0)
    :type change_interval: float
    :param lower_value: Lower extent of data value changes, inclusive (default=-1.0)
    :type lower_value: float
    :param upper_value: Upper extent of data value changes, exclusive (default=1.0)
    :type upper_value: float
    :param start_time: Start of affected time (default=0.0)
    :type start_time: float
    :param stop_time: Stop of affected time (default=600.0)
    :type stop_time: float
    """
    def behavior(self) -> Tuple[NDArray[Shape["change_times"], np.dtype[np.number]], NDArray[Shape["var"], np.dtype[np.number]]]:
        """
        behavior(self) -> change_times, change_values
        | **Description**
        |   First calculates how many times the data changes ``=n``.
        |   Then draws ``n+1``[#]_ many uniformly random values in [lower_value, upper_value) ``=change_values``.
        |   Finally draws the ``n+1`` random times in [start_time, stop_time) ``=change_times``

        :return: change_times, change_values
        :rtype: `NDArray[float] <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_, `NDArray[float] <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        
        .. [#] It is n+1 as the first behaviour change sets in at start_time
        """
        numberOfLoadChanges: int = int((self.stop_time-self.start_time) / self.change_interval)
        var = np.random.uniform(self.lower_value, self.upper_value, (numberOfLoadChanges + 1))
        change_time = np.insert(np.sort(np.random.uniform(self.start_time, self.stop_time, numberOfLoadChanges)), 0, [0])
        return change_time, var


@dataclass
class RandomTimeBrownBehavior(DataBehavior):
    """
    RandomTimeBrownBehavior(change_interval, lower_value, upper_value, start_time, stop_time)
    | **Description**
    |   Changes data behaviour in random intervals by a uniformly random value between lower_value and upper_value

    :param change_interval: Indicator[#]_ of time steps between behaviour changes (default=5.0)
    :type change_interval: float
    :param lower_value: Indicator of lower extent of data value changes, inclusive (default=-1.0)
    :type lower_value: float
    :param upper_value: Indicator of upper extent of data value changes, exclusive (default=1.0)
    :type upper_value: float
    :param start_time: Start of affected time (default=0.0)
    :type start_time: float
    :param stop_time: Stop of affected time (default=600.0)
    :type stop_time: float

    .. [#] The effect of the parameter is too complex to be simply described, 
        |  see `behavior` for the exact effect of the parameter. 
    """
    def behavior(self) -> Tuple[NDArray[Shape["change_times"], np.dtype[np.number]], NDArray[Shape["kps"], np.dtype[np.number]]]:
        """
        behavior(self) -> change_times, change_values
        | **Description**
        |   Calculates how many times the data changes ``=n``.
        |   Calculates an the average between lower_value and upper_Value ``=offset``.
        |   Draws n/4 random values in [-offset, offset) ``=observation_values``.
        |   Interpolates the observation_values through Gaussian Regression with Brownian Motion ``=interpolated_values``.
        |   Draws n*4 random times in [start_time, stop_time) ``=change_times``.
        |   Reads the values from interpolated_values at the change_times and adds the offset ``=change_values``
        

        :return: change_times, change_values
        :rtype: `NDArray[float] <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_, `NDArray[float] <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        """
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
        kp = kp[:,0,0]+offset
        return change_time, kp

