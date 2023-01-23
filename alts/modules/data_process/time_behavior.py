from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from alts.core.data_process.time_behavior import TimeBehavior
from alts.core.oracle.data_source import DataSource
from alts.core.data.constrains import QueryConstrain, ResultConstrain

if TYPE_CHECKING:
    from nptyping import NDArray, Shape, Number
    from typing import Tuple

@dataclass
class DataSourceTimeBehavior(TimeBehavior):

    data_source: DataSource

    def __post_init__(self):
        self.data_source = self.data_source()

    def query(self, times: NDArray[Shape["time_step_nr, [time]"], Number]) -> Tuple[NDArray[Shape["time_step_nr, [time]"], Number], NDArray[Shape["time_step_nr, ... var_shape"], Number]]:
        return self.data_source.query(times)
    
    @property
    def result_constrain(self) -> ResultConstrain:
        return self.data_source.result_constrain
    
    @property
    def query_constrain(self) -> QueryConstrain:
        return self.data_source.query_constrain

class NoTimeBehavior(TimeBehavior):

    def query(self, times: NDArray[Shape["time_step_nr, [time]"], Number]) -> Tuple[NDArray[Shape["time_step_nr, [time]"], Number], NDArray[Shape["time_step_nr, ... var_shape"], Number]]:
        return times, np.empty((times.shape[0],0))

    @property
    def result_constrain(self) -> ResultConstrain:
        return ResultConstrain((0,))

    @property
    def query_constrain(self) -> QueryConstrain:
        return QueryConstrain(count=None,shape=(1,),ranges=np.asarray([[0., np.inf]]))