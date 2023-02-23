from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

from alts.core.stopping_criteria import StoppingCriteria
from alts.core.configuration import init

if TYPE_CHECKING:
    from typing import Tuple, List

@dataclass
class TimeStoppingCriteria(StoppingCriteria):
    stop_time: float = init()

    @property
    def next(self) -> bool:
        return  self.stop_time >= self.exp.time_source.time

@dataclass
class DataExhaustedStoppingCriteria(StoppingCriteria):

    @property
    def next(self) -> bool:
        return False #TODO