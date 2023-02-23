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

    def __post_init__(self):
        super().__post_init__()
        self.data_source = self.data_source()

    def query(self, queries: NDArray[Shape["query_nr, ... query_shape"], Number]) -> Tuple[NDArray[Shape["query_nr, ... query_shape"], Number], NDArray[Shape["query_nr, ... result_shape"], Number]]:
        times = self.stream_data_pool.last_queries
        vars = self.stream_data_pool.last_results
        actual_queries = np.concatenate((times, vars, queries[:,2:]), axis=1)
        queries, results = self.data_source.query(actual_queries)
        self.last_queries = queries
        self.last_results = results
        self.ready = False
        return queries, results

    def update(self):
        queries = np.concatenate((self.stream_data_pool.last_queries, self.stream_data_pool.last_results, self.last_queries[:, 2:]), axis=1)
        queries, results = self.data_source.query(queries)
        if not self.ready:
            self.has_new_data = True
        return queries, results #return GT, as if queried

    def delayed_results(self) -> Tuple[NDArray[Shape["data_nr, ... query_shape"], Number], NDArray[Shape["data_nr, ... result_shape"], Number]]:
        self.has_new_data = False
        self.ready = True
        queries = np.concatenate((self.last_queries[:,:1] + self.time_source.time_step ,self.last_queries[:,1:]), axis=1) 
        return queries, self.last_results

    @property
    def query_constrain(self) -> QueryConstrain:
        return self.data_source.query_constrain

    @property
    def result_constrain(self) -> ResultConstrain:
        return self.data_source.result_constrain

    @property
    def delayed_constrain(self) -> ResultConstrain:
        return self.data_source.result_constrain

@dataclass
class IntegratingDSProcess(DataSourceProcess):

    integration_time: float = init(default=4)
    integrated_result: Optional[NDArray[Shape["data_nr, ... output_shape"], Number]] = pre_init(None)
    start_time: float = pre_init(0.0)
    end_time: float = pre_init(0.0)

    sliding_window: Optional[NDArray[Shape["data_nr, ... output_shape"], Number]] = pre_init(None)

    def query(self, queries: NDArray[Shape["query_nr, ... query_shape"], Number]) -> Tuple[NDArray[Shape["query_nr, ... query_shape"], Number], NDArray[Shape["query_nr, ... result_shape"], Number]]:
        queries, results = super().query(queries)
        self.start_time = queries[-1, 0]
        self.integrated_result = np.zeros_like(results)
        return queries, results

    def update(self):
        queries, results = super().update()
        self.has_new_data = False

        if self.integrated_result is not None:
            self.integrated_result = self.integrated_result + results

            time = self.time_source.time
            if self.start_time + self.integration_time <= time:
                self.end_time = time
                self.has_new_data = True

        return queries, results

    def delayed_results(self) -> Tuple[NDArray[Shape["data_nr, ... query_shape"], Number], NDArray[Shape["data_nr, ... result_shape"], Number]]:
        last_queries, last_results = super().delayed_results()
        integrated_result = is_set(self.integrated_result)
        self.integrated_result = None
        queries = np.concatenate((last_queries[:,:1] + (self.end_time - self.start_time),last_queries[:,1:]), axis=1) 
        return queries, integrated_result


@dataclass
class WindowDSProcess(DataSourceProcess):

    window_size: float = init(default=4)
    
    def __post_init__(self):
        super().__post_init__()
        self.sliding_query_window = np.empty((0, *self.data_source.query_constrain.shape))
        self.sliding_result_window = np.empty((0, *self.data_source.result_constrain.shape))

    def query(self, queries: NDArray[Shape["query_nr, ... query_shape"], Number]) -> Tuple[NDArray[Shape["query_nr, ... query_shape"], Number], NDArray[Shape["query_nr, ... result_shape"], Number]]:
        queries, results = super().query(queries)
        if not self.sliding_query_window.shape[0] == 0:
            queries = self.sliding_query_window[:1]
        return queries, results
    
    def update(self):
        queries, results = super().update()

        self.sliding_query_window = np.concatenate((self.sliding_query_window, queries))
        self.sliding_query_window = self.sliding_query_window[-self.window_size:]

        self.sliding_result_window = np.concatenate((self.sliding_result_window, results))
        self.sliding_result_window = self.sliding_result_window[-self.window_size:]

        queries = self.sliding_query_window[:1]
        results = np.sum(self.sliding_result_window, axis=0)[None,...]

        return queries, results

    def delayed_results(self) -> Tuple[NDArray[Shape["data_nr, ... query_shape"], Number], NDArray[Shape["data_nr, ... result_shape"], Number]]:
        last_queries, last_results = super().delayed_results()
        results = np.sum(self.sliding_result_window, axis=0)[None,...]
        return last_queries, results