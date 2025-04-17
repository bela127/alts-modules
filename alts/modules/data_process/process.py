#TODO correct all 
from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from dataclasses import dataclass, field

import numpy as np

from alts.core.oracle.data_source import DataSource
from alts.core.data_process.process import Process
from alts.core.configuration import is_set, post_init, pre_init, init, NOTSET
from alts.core.data.constrains import QueryConstrain, ResultConstrain
from alts.modules.oracle.data_source import TimeBehaviorDataSource
from alts.modules.behavior import RandomTimeUniformBehavior
from alts.core.subscriber import TimeSubscriber, ProcessOracleSubscriber
from alts.core.data.constrains import DelayedConstrained
from alts.core.data.data_pools import StreamDataPools, ResultDataPools, PRDataPools, SPRDataPools
from alts.core.oracle.oracles import POracles

if TYPE_CHECKING:
    from typing_extensions import Self
    from typing import Tuple
    from nptyping import NDArray, Number, Shape
    from alts.core.configuration import Required
    from alts.core.oracle.data_source import TimeDataSource

@dataclass
class StreamProcess(Process, TimeSubscriber):
    """
    StreamProcess(time_source, data_pools, oracles, stop_time, time_behaviour)
    | **Description**
    |   StreamProcess is a process for data streams.

    :param time_source: Source of time
    :type time_source: :doc:`TimeSource </core/data_process/process>`
    :param data_pools: A data structure which saves all processed queries and results
    :type data_pools: :doc:`DataPools </core/data/data_pools>`
    :param oracles: The interaction point between the Process and the data source.
    :type oracles: :doc:`Oracles </core/oracle/oracles>`
    :param stop_time: The stopping time of the experiment (default= 1000)
    :type stop_time: float
    :param time_behaviour: A DataSource with time-dependent data
    :type time_behaviour: TimeDataSource
    """
    stop_time: float = init(default=1000)
    time_behavior: TimeDataSource = init()

    data_pools: StreamDataPools = post_init()

    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Initializes its :doc:`TimeDataSource </core/data_process/time_source>` and :doc:`StreamDataPools </core/data/data_pools>`. 
        
        :raises TypeError: If the DataPools is not a StreamDataPools
        """
        if self.time_behavior is NOTSET:
            self.time_behavior = TimeBehaviorDataSource(behavior=RandomTimeUniformBehavior(stop_time=self.stop_time))
        super().post_init()
        
        self.time_behavior = self.time_behavior()
        if isinstance(self.data_pools, StreamDataPools):
            self.data_pools.stream = self.data_pools.stream(query_constrain = self.time_behavior.query_constrain, result_constrain=self.time_behavior.result_constrain)
        else:
            raise TypeError(f"StreamProcess requires StreamDataPools")

    def time_update(self, subscription):
        """
        time_update(self, subscription) -> times, vars
        | **Description**
        |   Returns the current time and its corresponding result.

        :param subscription: Unused
        :type subscription: Any
        :return: Current time, Current result
        :rtype: Tuple[`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_, `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_]
        """
        times = np.asarray([[self.time_source.time]])
        times, vars = self.time_behavior.query(times)
        self.data_pools.stream.add((times, vars))
        return times, vars

@dataclass
class DataSourceProcess(Process, ProcessOracleSubscriber):
    """
    DataSourceProcess(time_source, data_pools, oracles, data_source)
    | **Description**
    |   DataSourceProcess is a process specifically for data sources.

    :param time_source: Source of time
    :type time_source: :doc:`TimeSource </core/data_process/process>`
    :param data_pools: A data structure which saves all processed queries and results
    :type data_pools: :doc:`ResultDataPools </core/data/data_pools>`
    :param oracles: The interaction point between the Process and the data source.
    :type oracles: :doc:`Oracles </core/oracle/oracles>`
    :param data_source: A DataSource containing all data points
    :type data_source: DataSource
    """
    data_source: DataSource = init()

    data_pools: ResultDataPools = post_init()
    oracles: POracles = post_init()

    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Initializes its DataSource, POracles and :doc:`ResultDataPools </core/data/data_pools>`. 
        
        :raises TypeError: If the DataPools is not a ResultDataPools or the oracles is not POracles
        """
        self.data_source = self.data_source()
        if isinstance(self.oracles, POracles):
            self.oracles.process = self.oracles.process(query_constrain=self.query_constrain)
        else:
            raise TypeError(f"DataSourceProcess requires POracles")
        super().post_init()
        if isinstance(self.data_pools, ResultDataPools):
            self.data_pools.result = self.data_pools.result(query_constrain=self.query_constrain, result_constrain=self.result_constrain)
        else:
            raise TypeError(f"DataSourceProcess requires ResultDataPools")

    
    def process_query(self, subscription):
        """
        process_query(self, subscription) -> None
        | **Description**
        |   Pops all processed queries in the oracle and adds them to its own data pools. 

        :param subscription: Does nothing
        :type subscription: Any
        """
        queries = self.oracles.process.pop(self.oracles.process.count)
        queries, results = self.query(queries)
        self.data_pools.result.add((queries, results))
    

    def query(self, queries: NDArray[Shape["query_nr, ... query_shape"], Number]) -> Tuple[NDArray[Shape["query_nr, ... query_shape"], Number], NDArray[Shape["query_nr, ... result_shape"], Number]]: # type: ignore
        """
        query(self, queries) -> data_points
        | **Description**
        |   Returns all given queries with their associated results from the data source.

        :param queries: Requested queries
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Requested queries and their associated results.
        :rtype: Tuple[`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_,`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_]
        """
        queries, results = self.data_source.query(queries)
        return queries, results

    def query_constrain(self) -> QueryConstrain:
        """
        query_constrain(self) -> QueryConstrain
        | **Description**
        |   Returns its own query constraints.

        :return: Data pool's query constraints
        :rtype: QueryConstrain
        """
        return self.data_source.query_constrain()

    def result_constrain(self) -> ResultConstrain:
        """
        result_constrain(self) -> ResultConstrain
        | **Description**
        |   Returns its own result constraints.

        :return: Data pool's result constraints
        :rtype: ResultConstrain
        """
        return self.data_source.result_constrain()


@dataclass
class DelayedProcess(Process, DelayedConstrained):
    """
    DelayedProcess(time_source, data_pools, oracles, data_source)
    | **Description**
    |   DelayedProcess is a process specifically for data where queries have intermediate and time-delayed results.

    :param time_source: Source of time
    :type time_source: :doc:`TimeSource </core/data_process/process>`
    :param data_pools: A data structure which saves all processed queries and results
    :type data_pools: :doc:`ResultDataPools </core/data/data_pools>`
    :param oracles: The interaction point between the Process and the data source.
    :type oracles: :doc:`Oracles </core/oracle/oracles>`
    :param data_source: A DataSource containing all data points
    :type data_source: DataSource
    """
    data_source: DataSource = init()

    has_new_data: bool = pre_init(default=False)
    ready: bool = pre_init(default=True)

    data_pools: PRDataPools = post_init()
    oracles: POracles = post_init()

    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Initializes its DataSource, POracles and :doc:`ResultDataPools </core/data/data_pools>`. 
        
        :raises TypeError: If the DataPools is not a ResultDataPools or the oracles is not POracles
        """
        super().post_init()
        self.data_source = self.data_source()

        if isinstance(self.data_pools, ResultDataPools):
            self.data_pools.process = self.data_pools.process(query_constrain=self.query_constrain, result_constrain=self.result_constrain)
            self.data_pools.result = self.data_pools.result(query_constrain=self.query_constrain, result_constrain=self.delayed_constrain)
        else:
            raise TypeError(f"DelayedProcess requires PRDataPools")

        if isinstance(self.oracles, POracles):
            self.oracles.process = self.oracles.process(query_constrain=self.query_constrain)
        else:
            raise TypeError(f"DataSourceProcess requires POracles")

    
    def step(self, iteration): #TODO
        """
        step(self, iteration) -> Tuple[data_points, delayed_data_points]
        | **Description**
        |

        :param iteration:
        :type iteration:
        :return: 
        :rtype:
        """
        queries, results = self.add_intermediate_results()

        self.update()
        
        delayed_queries, delayed_results = self.add_results()
        return queries, results, delayed_queries, delayed_results
    
    def add_intermediate_results(self):
        """
        add_intermediate_results(self) -> data_points
        | **Description**
        |

        :param :
        :type :
        :return:
        :rtype:
        """
        queries = None
        results = None
        if not self.oracles.process.empty and self.ready:
            queries = self.oracles.process.pop()
            queries, results = self.query(queries)
            self.added_intermediate_data(queries, results)
            self.data_pools.process.add((queries, results))
        return queries, results

    def add_results(self):
        delayed_queries = None
        delayed_results = None
        if self.has_new_data:
            delayed_queries, delayed_results = self.delayed_query()
            self.added_data(delayed_queries, delayed_results)
            self.data_pools.result.add((delayed_queries, delayed_results))
        return delayed_queries, delayed_results

    def query(self, queries: NDArray[Shape["query_nr, ... query_shape"], Number]) -> Tuple[NDArray[Shape["query_nr, ... query_shape"], Number], NDArray[Shape["query_nr, ... result_shape"], Number]]:
        #Bug? ResultDataPools has no stream attribute
        times = self.data_pools.stream.last_queries
        vars = self.data_pools.stream.last_results
        actual_queries = np.concatenate((times, vars, queries[:,2:]), axis=1)
        queries, results = self.data_source.query(actual_queries)
        return queries, results
    
    def added_intermediate_data(self, queries, results):
        self.last_queries = queries
        self.last_results = results
        self.ready = False

    def update(self):
        queries = np.concatenate((self.data_pools.stream.last_queries, self.data_pools.stream.last_results, self.last_queries[:, 2:]), axis=1)
        queries, results = self.data_source.query(queries)
        if not self.ready:
            self.has_new_data = True
        return queries, results #return GT, as if queried
    
    def added_data(self, queries, results):
        """
        added_data(self. queries, results) -> None
        | **Description**
        |   Informs this process that all new data has been added.

        :param queries: Unused 
        :type queries: Any
        :param results: Unused
        :type results: Any
        """
        self.has_new_data = False
        self.ready = True

    def delayed_query(self) -> Tuple[NDArray[Shape["data_nr, ... query_shape"], Number], NDArray[Shape["data_nr, ... result_shape"], Number]]: # type: ignore
        """
        delayed_query(self) -> data_points
        | **Description**
        |

        :return:
        :rtype:
        """
        queries = np.concatenate((self.last_queries[:,:1] + self.time_source.time_step ,self.last_queries[:,1:]), axis=1) 
        return queries, self.last_results

    def query_constrain(self) -> QueryConstrain:
        """
        query_constrain(self) -> QueryConstrain
        | **Description**
        |   Returns its own query constraints.

        :return: Data pool's query constraints
        :rtype: QueryConstrain
        """
        return self.data_source.query_constrain()

    def result_constrain(self) -> ResultConstrain:
        """
        result_constrain(self) -> ResultConstrain
        | **Description**
        |   Returns its own result constraints.

        :return: Data pool's result constraints
        :rtype: ResultConstrain
        """
        return self.data_source.result_constrain()

    def delayed_constrain(self) -> ResultConstrain:
        """
        delayed_constrain(self) -> ResultConstrain
        | **Description**
        |   Returns its own delayed result constraints.

        :return: Data pool's delayed result constraints
        :rtype: ResultConstrain
        """
        return self.data_source.result_constrain()

@dataclass
class DelayedStreamProcess(DelayedProcess, StreamProcess):
    pass

@dataclass
class IntegratingDSProcess(DelayedStreamProcess):
    #QUESTION what does it do?
    integration_time: float = init(default=4)
    integrated_result: Optional[NDArray[Shape["data_nr, ... output_shape"], Number]] = pre_init(None)
    start_time: float = pre_init(0.0)
    end_time: float = pre_init(0.0)

    sliding_window: Optional[NDArray[Shape["data_nr, ... output_shape"], Number]] = pre_init(None)
    
    def added_intermediate_data(self, queries, results):
        super().added_intermediate_data(queries, results)
        self.start_time = queries[-1, 0]
        self.integrated_result = np.zeros_like(results)

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
    
    def added_data(self, queries, results):
        super().added_data(queries, results)
        self.integrated_result = None

    def delayed_query(self) -> Tuple[NDArray[Shape["data_nr, ... query_shape"], Number], NDArray[Shape["data_nr, ... result_shape"], Number]]:
        last_queries, last_results = super().delayed_query()
        integrated_result = is_set(self.integrated_result)
        queries = np.concatenate((last_queries[:,:1] + (self.end_time - self.start_time),last_queries[:,1:]), axis=1) 
        return queries, integrated_result


@dataclass
class WindowDSProcess(DelayedStreamProcess):

    window_size: float = init(default=4)
    
    def post_init(self):
        super().post_init()
        self.sliding_query_window = np.empty((0, *self.data_source.query_constrain().shape))
        self.sliding_result_window = np.empty((0, *self.data_source.result_constrain().shape))
    
    def added_intermediate_data(self, queries, results):
        super().added_intermediate_data(queries, results)
        if not self.sliding_query_window.shape[0] == 0:
            queries = self.sliding_query_window[:1]
    
    def update(self):
        queries, results = super().update()

        self.sliding_query_window = np.concatenate((self.sliding_query_window, queries))
        self.sliding_query_window = self.sliding_query_window[-self.window_size:]

        self.sliding_result_window = np.concatenate((self.sliding_result_window, results))
        self.sliding_result_window = self.sliding_result_window[-self.window_size:]

        queries = self.sliding_query_window[:1]
        results = np.sum(self.sliding_result_window, axis=0)[None,...]

        return queries, results

    def delayed_query(self) -> Tuple[NDArray[Shape["data_nr, ... query_shape"], Number], NDArray[Shape["data_nr, ... result_shape"], Number]]:
        last_queries, last_results = super().delayed_query()
        results = np.sum(self.sliding_result_window, axis=0)[None,...]
        return last_queries, results