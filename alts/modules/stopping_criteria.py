#Version 1.1.1 conform as of 20.12.2024
"""
| *alts.modules.stopping_criteria*
| :doc:`Core Module </core/stopping_criteria>`
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

from alts.core.stopping_criteria import StoppingCriteria
from alts.core.configuration import init

from alts.modules.data_process.process import DataSourceProcess

if TYPE_CHECKING:
    from typing import Tuple, List

@dataclass
class TimeStoppingCriteria(StoppingCriteria):
    """
    TimeStoppingCriteria(stop_time)
    | **Description**
    |   This stopping criteria is fulfilled if the internal experiment time exceeds a given ``stop_time``.

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

        :return: True if experiment time has not reached or exceeded stopping time (else False)
        :rtype: bool
        """
        return  self.stop_time >= self.exp.time_source.time

@dataclass
class DataExhaustedStoppingCriteria(StoppingCriteria):
    """
    DataExhaustedStoppingCriteria()
    | **Description**
    |   Stops the experiment if the DataSource has been exhausted.
    """

    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Checks for compatability between this StoppingCriteria and the configured Process in the experiment.

        :raises TypeError: If the experiment's process is not a DataSourceProcess
        """
        super().post_init()
        if not isinstance(self.exp.process, DataSourceProcess):
            raise TypeError(f"DataExhaustedStoppingCriteria requires DataSourceProcess")

    @property
    def next(self) -> bool:
        """
        next(self) -> bool
        | **Description**
        |   Checks whether the experiment should stop.

        :return: True if the experiment's DataSource has not been exhausted (else False)
        :rtype: bool
        """
        return not self.exp.process.data_source.exhausted # type: ignore