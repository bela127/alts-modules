#Version 1.1.1 conform as of 23.04.2025
"""
| *alts.modules.data_process.process*
| :doc:`Core Module </core/data_process/process>`
"""
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
    StreamProcess(stop_time, time_behaviour)
    | **Description**
    |   StreamProcess is a process for data streams.

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
    DataSourceProcess(data_source)
    | **Description**
    |   DataSourceProcess is a process specifically for data sources.

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
    DelayedProcess(data_source)
    | **Description**
    |   DelayedProcess is a process specifically for data where queries have intermediate and time-delayed results.

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
 
    def step(self, iteration):
        """
        step(self, iteration) -> Tuple[data_points, delayed_data_points]
        | **Description**
        |   Processes the next queries in the queue. Returns their intermediate and delayed results.

        :param iteration: Unused
        :type iteration: Any
        :return: Intermediate queries, their results, Delayed queries, their results
        :rtype: Tuple[Tuple[`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_,`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_],Tuple[`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_,`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_]]
        """
        queries, results = self.add_intermediate_results()

        self.update()
        
        delayed_queries, delayed_results = self.add_results()
        return queries, results, delayed_queries, delayed_results
    
    def add_intermediate_results(self):
        """
        add_intermediate_results(self) -> data_points
        | **Description**
        |   Processes the next queries and returns the intermediate results.

        :return: Processed intermediate queries and their associated results
        :rtype: Tuple[`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_,`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_]
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
        """
        add_results(self) -> delayed_data_points
        | **Description**
        |   If an intermediate query has been made, queries the delayed queries and results and adds them to its DataPools.

        :return: Delayed data points
        :rtype: Tuple[`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_,`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_]
        """
        delayed_queries = None
        delayed_results = None
        if self.has_new_data:
            delayed_queries, delayed_results = self.delayed_query()
            self.added_data(delayed_queries, delayed_results)
            self.data_pools.result.add((delayed_queries, delayed_results))
        return delayed_queries, delayed_results

    def query(self, queries: NDArray[Shape["query_nr, ... query_shape"], Number]) -> Tuple[NDArray[Shape["query_nr, ... query_shape"], Number], NDArray[Shape["query_nr, ... result_shape"], Number]]: # type: ignore
        #TODO Bug? ResultDataPools has no stream attribute
        """
        query(queries) -> data_points
        | **Description**
        |   Queries the given queries and returns their data points.

        :param queries: Requested queries
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Queried queries, Queried queries' results
        :rtype: Tuple[`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_,`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_]
        """
        times = self.data_pools.stream.last_queries
        vars = self.data_pools.stream.last_results
        actual_queries = np.concatenate((times, vars, queries[:,2:]), axis=1)
        queries, results = self.data_source.query(actual_queries)
        return queries, results
    
    def added_intermediate_data(self, queries, results):
        """
        added_intermediate_data(self, queries, results) -> None
        | **Description**
        |   Updates its last processed queries and results, and sets its status not ready to process new queries.

        :param queries: Last processed intermediate queries
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :param results: Last processed queries' delayed results
        :type results: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        """
        self.last_queries = queries
        self.last_results = results
        self.ready = False

    def update(self):
        """
        update(self) ->  data_points
        | **Description**
        |   Queries the last queried queries again. Sets the ```has_new_data``` flag to True and returns the data points.

        :return: Intermediate queries, delayed results
        :rtype: Tuple[`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_,`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_]
        """
        #TODO Bug? ResultDataPools has no stream attribute
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
        |   Returns the time-modified delayed queries with the delayed results.

        :return: Delayed queries, delayed results
        :rtype: Tuple[`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_,`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_]
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
    """
    DelayedStreamProcess(data_source, stop_time, time_behavior)
    | **Description**
    |   A DelayedProcess for StreamProcesses.

    :param data_source: A DataSource containing all data points
    :type data_source: DataSource
    :param stop_time: The stopping time of the experiment (default= 1000)
    :type stop_time: float
    :param time_behaviour: A DataSource with time-dependent data
    :type time_behaviour: TimeDataSource
    """
    pass

@dataclass
class IntegratingDSProcess(DelayedStreamProcess):
    """
    IntegratingDSProcess(data_source, stop_time, time_behavior)
    | **Description**
    |   The Integrating Delayed Stream Process integrates its results over a given time period.

    :param data_source: A DataSource containing all data points
    :type data_source: DataSource
    :param stop_time: The stopping time of the experiment (default= 1000)
    :type stop_time: float
    :param time_behaviour: A DataSource with time-dependent data
    :type time_behaviour: TimeDataSource
    :param integration_time: What time period to integrate the results over (default= 4)
    :type integration_time: float
    """
    integration_time: float = init(default=4)
    integrated_result: Optional[NDArray[Shape["data_nr, ... output_shape"], Number]] = pre_init(None) # type: ignore
    start_time: float = pre_init(0.0)
    end_time: float = pre_init(0.0)

    sliding_window: Optional[NDArray[Shape["data_nr, ... output_shape"], Number]] = pre_init(None) # type: ignore
    
    def added_intermediate_data(self, queries, results):
        """
        added_intermediate_data(self, queries, results) -> None
        | **Description**
        |   Updates its last processed queries and results, sets its status not ready to process new queries, sets start time and sets the integrated result to a zero-matrix.

        :param queries: Last processed intermediate queries
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :param results: Last processed queries' delayed results
        :type results: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        """
        super().added_intermediate_data(queries, results)
        self.start_time = queries[-1, 0]
        self.integrated_result = np.zeros_like(results)

    def update(self):
        """
        update(self) ->  data_points
        | **Description**
        |   Queries the last queried queries again. Sets the end time and ```has_new_data``` flag to True if the integration time has passed. Adds the result to the integrated result. Finally, returns the data points.

        :return: Intermediate queries, delayed results
        :rtype: Tuple[`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_,`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_]
        """
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
        """
        added_data(self. queries, results) -> None
        | **Description**
        |   Informs this process that all new data has been added and unassigns the integrated result.

        :param queries: Unused 
        :type queries: Any
        :param results: Unused
        :type results: Any
        """
        super().added_data(queries, results)
        self.integrated_result = None

    def delayed_query(self) -> Tuple[NDArray[Shape["data_nr, ... query_shape"], Number], NDArray[Shape["data_nr, ... result_shape"], Number]]: # type: ignore
        """
        delayed_query(self) -> data_points
        | **Description**
        |   Returns the time-modified delayed queries and the unchanged integrated result.

        :return: Delayed queries, integrated results
        :rtype: Tuple[`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_,`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_]
        :raises: ValueError if integrated_result is None
        """
        last_queries, last_results = super().delayed_query()
        integrated_result = is_set(self.integrated_result)
        queries = np.concatenate((last_queries[:,:1] + (self.end_time - self.start_time),last_queries[:,1:]), axis=1) 
        return queries, integrated_result


@dataclass
class WindowDSProcess(DelayedStreamProcess):
    """
    WindowDSProcess(data_source, stop_time, time_behavior, window_size)
    | **Description**
    |   The Window Delayed Stream Process integrates the results over a given window size.
    |   Similar to IntegratingDSProcess, where a new window is opened and filled, except here the window slides over with each new query, thus always being filled.

    :param data_source: A DataSource containing all data points
    :type data_source: DataSource
    :param stop_time: The stopping time of the experiment (default= 1000)
    :type stop_time: float
    :param time_behaviour: A DataSource with time-dependent data
    :type time_behaviour: TimeDataSource
    :param window_size: The size of the integrating window (default= 4)
    :type window_size: float
    """
    window_size: float = init(default=4)
    
    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Initializes its DataSource, POracles, :doc:`ResultDataPools </core/data/data_pools>` and sliding windows. 
        
        :raises TypeError: If the DataPools is not a ResultDataPools or the oracles is not POracles
        """
        super().post_init()
        self.sliding_query_window = np.empty((0, *self.data_source.query_constrain().shape))
        self.sliding_result_window = np.empty((0, *self.data_source.result_constrain().shape))
    
    def added_intermediate_data(self, queries, results):
        """
        added_intermediate_data(self, queries, results) -> None
        | **Description**
        |   Updates its last processed queries and results, and sets its status not ready to process new queries.
        |   Also removes the first query of the sliding query window.

        :param queries: Last processed intermediate queries
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :param results: Last processed queries' delayed results
        :type results: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        """
        super().added_intermediate_data(queries, results)
        if not self.sliding_query_window.shape[0] == 0:
            queries = self.sliding_query_window[:1]
    
    def update(self):
        """
        update(self) ->  data_points
        | **Description**
        |   Queries the last queried queries again. Sets the ```has_new_data``` flag to True and returns the data points.
        |   Also updates the sliding query and result windows.

        :return: Intermediate queries, delayed results
        :rtype: Tuple[`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_,`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_]
        """
        queries, results = super().update()

        self.sliding_query_window = np.concatenate((self.sliding_query_window, queries))
        self.sliding_query_window = self.sliding_query_window[-self.window_size:]

        self.sliding_result_window = np.concatenate((self.sliding_result_window, results))
        self.sliding_result_window = self.sliding_result_window[-self.window_size:]

        queries = self.sliding_query_window[:1]
        results = np.sum(self.sliding_result_window, axis=0)[None,...]

        return queries, results

    def delayed_query(self) -> Tuple[NDArray[Shape["data_nr, ... query_shape"], Number], NDArray[Shape["data_nr, ... result_shape"], Number]]: # type: ignore
        """
        delayed_query(self) -> data_points
        | **Description**
        |   Returns the time-modified delayed queries with the sum of the sliding result window.

        :return: Delayed queries, Sliding result window sum
        :rtype: Tuple[`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_,`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_]
        """
        last_queries, last_results = super().delayed_query()
        results = np.sum(self.sliding_result_window, axis=0)[None,...]
        return last_queries, results