from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from dataclasses import dataclass, field

from alts.core.oracle.data_source import DataSource
from alts.core.data_process.process import Process
import numpy as np
from alts.core.configuration import is_set, ConfAttr, post_init, pre_init, init
from alts.core.data.constrains import QueryConstrain, ResultConstrain

if TYPE_CHECKING:
    from typing_extensions import Self
    from typing import Tuple
    from nptyping import NDArray, Number, Shape
    from alts.core.configuration import Required
    from alts.core.data_process.time_source import TimeSource

@dataclass
class DataSourceProcess(Process):

    data_source: DataSource = init()
    _data_points: Tuple[NDArray[Shape["data_nr, ... query_shape"], Number], NDArray[Shape["data_nr, ... output_shape"], Number]] = post_init()
    _new_data: bool = pre_init(default=False)

    def __post_init__(self):
        super().__post_init__()
        self.data_source = self.data_source()

    def run(self, times, vars) -> Tuple[NDArray[Shape["data_nr, ... query_shape"], Number], NDArray[Shape["data_nr, ... output_shape"], Number]]:
        queries = self.query_queue.pop()
        self._data_points =  self.data_source.query(queries)
        self._new_data = True
        return self._data_points

    @property
    def finished(self) -> bool:
        new_data = self._new_data
        self._new_data = False
        return new_data

    @property
    def results(self) -> Tuple[NDArray[Shape["data_nr, ... query_shape"], Number], NDArray[Shape["data_nr, ... result_shape"], Number]]:
        return self._data_points

    @property
    def query_constrain(self) -> QueryConstrain:
        return self.data_source.query_constrain

    @property
    def result_constrain(self) -> ResultConstrain:
        return self.data_source.result_constrain

@dataclass
class TimeVariantDSProcess(DataSourceProcess):

    def run(self, times, vars) -> Tuple[NDArray[Shape["data_nr, ... query_shape"], Number], NDArray[Shape["data_nr, ... output_shape"], Number]]:
        queries = self.query_queue.pop()
        actual_queries = np.concatenate((times, queries, vars), axis=1)
        data_points =  self.data_source.query(actual_queries)
        extended_queries, results = data_points
        queries = extended_queries[:, times.shape[1]:-vars.shape[1]]
        self._data_points = (queries, results)
        self._new_data = True
        return self._data_points
    
    @property
    def query_constrain(self) -> QueryConstrain:

        ds_qc: QueryConstrain = self.data_source.query_constrain
        dim = ds_qc.shape[0] - self.var_constrain.shape[0] - self.time_source.result_constrain.shape[0]
        ranges = ds_qc.ranges[self.time_source.result_constrain.shape[0]: self.var_constrain.shape[0]]
        query_constrain = QueryConstrain(count=ds_qc.count, shape=(dim,),ranges=ranges)

        return query_constrain


@dataclass
class IntegratingDSProcess(TimeVariantDSProcess):

    integration_time: float = init()
    integrated_result: Optional[NDArray[Shape["data_nr, ... output_shape"], Number]] = None
    start_time = 0

    def run(self, times, vars) -> Tuple[NDArray[Shape["data_nr, ... query_shape"], Number], NDArray[Shape["data_nr, ... output_shape"], Number]]:
        queries, results = super().run(times, vars)
        self._new_data = False
        if self.integrated_result is None:
            self.start_time = times[0]
            self.integrated_result = results
        else:
            self.integrated_result = self.integrated_result + results
        return self._data_points

    @property
    def finished(self) -> bool:
        if self.start_time + self.integration_time >= self.time_source.time:
            self._new_data = True
        return super().finished

    @property
    def results(self) -> Tuple[NDArray[Shape["data_nr, ... query_shape"], Number], NDArray[Shape["data_nr, ... result_shape"], Number]]:
        integrated_result = is_set(self.integrated_result)
        self.integrated_result = None
        queries, results = self._data_points
        return queries, integrated_result