#Version 1.1.1 conform as of 20.12.2024
"""
| *alts.modules.stopping_criteria*
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

from alts.core.stopping_criteria import StoppingCriteria
from alts.core.configuration import init

if TYPE_CHECKING:
    from typing import Tuple, List

@dataclass
class TimeStoppingCriteria(StoppingCriteria):
    """
    TimeStoppingCriteria(exp, stop_time)
    | **Description**
    |   This stopping criteria is fulfilled if the internal experiment time exceeds a given ``stop_time``.

    :param exp: The experiment to monitor
    :type exp: Experiment
    :param stop_time: The stopping time
    :type stop_time: float
    """
    stop_time: float = init()

    @property
    def next(self) -> bool:
        """
        next(self) -> bool
        | **Description**
        |   Checks whether the experiment should stop.

        :return: True if experiment time has reached or exceeded stopping time (else False)
        :rtype: bool
        """
        return  self.stop_time >= self.exp.time_source.time

@dataclass
class DataExhaustedStoppingCriteria(StoppingCriteria):
    """
    DataExhaustedStoppingCriteria(exp)
    | **Description**
    |

    :param exp: The experiment to monitor
    :type exp: Experiment
    """
    @property
    def next(self) -> bool:
        """
        next(self) -> bool
        | **Description**
        |   Checks whether the experiment should stop.

        :return: True if the experiment's DataSource has been exhausted (else False)
        :rtype: bool
        """
        return False #TODO