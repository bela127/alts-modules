#Version 1.1.1 conform as of 23.04.2025
"""
| *alts.modules.evaluator*
| :doc:`Core Module </core/evaluator>`
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field
import os
import time


from alts.core.evaluator import Evaluator, Evaluate, LogingEvaluator
from alts.core.data.data_pools import StreamDataPools, ProcessDataPools, ResultDataPools
from alts.core.oracle.oracles import POracles
from alts.core.configuration import pre_init, post_init
from alts.modules.data_process.process import DelayedProcess

import numpy as np
from matplotlib import pyplot as plot # type: ignore


if TYPE_CHECKING:
    from alts.core.experiment import Experiment
    from nptyping import  NDArray, Number, Shape
    from alts.core.oracle.data_source import DataSource


class PrintNewDataPointsEvaluator(Evaluator):
    """
    PrintNewDataPointsEvaluator()
    | **Description**
    |   This evaluator keeps track of the experiment's results.

    """
    def register(self, experiment: Experiment):
        """
        register(self, experiment) -> None
        | **Description**
        |   Modifies the experiment to print new data points before adding them to the experiment's result data pool.
        |   Requires the experiment's data pool to be a ResultDataPool.

        :param experiment: The experiment to be evaluated
        :type experiment: Experiment
        :raises: TypeError if self.experiment.data_pools is not a ResultDataPools
        """
        super().register(experiment)

        if isinstance(self.experiment.data_pools, ResultDataPools):
            self.experiment.data_pools.result.add = Evaluate(self.experiment.data_pools.result.add)
            self.experiment.data_pools.result.add.pre(self._print_new_data_points)
        else:
            raise TypeError(f"PrintNewDataPointsEvaluator requires ResultDataPools")

    def _print_new_data_points(self, data_points):
        """
        _print_new_data_points(self, data_points) -> None
        | **Description**
        |   Prints the given data points.

        :param data_points: The data points to be printed
        :type data_points: Tuple[NDArray[Shape["query_nr, ... query_dim"], Number], NDArray[Shape["query_nr, ... result_dim"], Number]]
        """
        print(data_points)

class PrintQueryEvaluator(Evaluator):
    """
    PrintQueryEvaluator()
    | **Description**
    |   This evaluator keeps track of the experiment's queries.

    """
    def register(self, experiment: Experiment):
        """
        register(self, experiment) -> None
        | **Description**
        |   Modifies the experiment to print new queries before adding them to the experiment's query queue.
        |   Requires the experiment's oracle to be a POracles.

        :param experiment: The experiment to be evaluated
        :type experiment: Experiment
        :raises: TypeError if self.experiment.oracles is not a POracles
        """
        super().register(experiment)

        if isinstance(self.experiment.oracles, POracles):
            self.experiment.oracles.process.add = Evaluate(self.experiment.oracles.process.add)
            self.experiment.oracles.process.add.pre(self.print_query)
        else:
            raise TypeError(f"PrintQueryEvaluator requires POracles")

    def print_query(self, queries):
        """
        print_query(self, queries) -> None
        | **Description**
        |   Prints the given queries.

        :param queries: New queries going to the query queue
        :type queries: Tuple[NDArray[Shape["query_nr, ... query_dim"], Number]
        """
        print("Queried: \n",queries)

class PrintExpTimeEvaluator(Evaluator):
    """
    PrintExpTimeEvaluator()
    | **Description**
    |   This evaluator measures how long the experiment takes to run.

    """
    def register(self, experiment: Experiment):
        """
        register(self, experiment) -> None
        | **Description**
        |   Modifies the experiment to print how long it took to run.

        :param experiment: The experiment to be evaluated
        :type experiment: Experiment
        """
        super().register(experiment)

        self.experiment.run = Evaluate(self.experiment.run)
        self.experiment.run.pre(self.start_time)
        self.experiment.run.post(self.end_time)

    def start_time(self):
        """
        start_time(self) -> None
        | **Description**
        |   Prints an anouncement of time measurement. Saves the current time as the start time.
        """
        print(f"Start timing for {self.experiment.exp_name} {self.experiment.exp_nr}")
        self.start = time.time()
    
    def end_time(self, result):
        """
        end_time(self) -> None
        | **Description**
        |   Substracts the start time from the current time and prints the resulting time.
        """
        end = time.time()
        print(f"Time for {self.experiment.exp_name} {self.experiment.exp_nr}: ",end - self.start)

class PrintTimeSourceEvaluator(Evaluator):
    """
    PrintTimeSourceEvaluator()
    | **Description**
    |   This Evaluator keeps track of the experiment's internal time.

    """
    def register(self, experiment: Experiment):
        """
        register(self, experiment) -> None
        | **Description**
        |   Modifies the experiment's time source to print the current internal time at each time step.

        :param experiment: The experiment to be evaluated
        :type experiment: Experiment
        """
        super().register(experiment)

        self.experiment.time_source.step = Evaluate(self.experiment.time_source.step)
        self.experiment.time_source.step.post(self.end_time)
    
    def end_time(self, time):
        """
        end_time(self, time) -> None
        | **Description**
        |   Prints the given time.

        :param time: The experiment's current internal time
        :type time: int
        """
        print("Sim Unit Time: ", time)

@dataclass
class PlotNewDataPointsEvaluator(LogingEvaluator):
    """
    PlotNewDataPointsEvaluator()
    | **Description**
    |   This evaluator plots all of the experiment's new data points continuously as they arrive.

    """
    interactive: bool = False
    folder: str = "fig"
    fig_name:str = "Data"

    queries: NDArray[Shape["query_nr, ... query_dim"], Number] = pre_init(None) # type: ignore
    results: NDArray[Shape["query_nr, ... result_dim"], Number] = pre_init(None) # type: ignore

    def register(self, experiment: Experiment):
        """
        register(self, experiment) -> None
        | **Description**
        |   Modifies the experiment to plot new data points before adding them to the experiment's result data pool.
        |   Requires the experiment's data pool to be a ResultDataPool.

        :param experiment: The experiment to be evaluated
        :type experiment: Experiment
        :raises: TypeError if self.experiment.data_pools is not a ResultDataPools
        """
        super().register(experiment)

        os.makedirs(self.path, exist_ok=True)

        if isinstance(self.experiment.data_pools, ResultDataPools):
            self.experiment.data_pools.result.add = Evaluate(self.experiment.data_pools.result.add)
            self.experiment.data_pools.result.add.pre(self.plot_new_data_points)
        else:
            raise TypeError(f"PlotNewDataPointsEvaluator requires ResultDataPools")

        self.queries: NDArray[Shape["query_nr, ... query_dim"], Number] = None # type: ignore
        self.results: NDArray[Shape["query_nr, ... result_dim"], Number] = None # type: ignore

    def plot_new_data_points(self, data_points):
        """
        plot_new_data_points(self, data_points) -> None
        | **Description**
        |   Adds the given data points to the saved ones and updates the plot.

        :param data_points: New data points to be plotted
        :type data_points: Tuple[NDArray[Shape["query_nr, ... query_dim"], Number], NDArray[Shape["query_nr, ... result_dim"], Number]]
        """
        self.experiment.iteration
        queries, results = data_points

        if self.queries is None:
            self.queries = queries
            self.results = results
        else:
            self.queries = np.concatenate((self.queries, queries))
            self.results = np.concatenate((self.results, results))

        fig = plot.figure(self.fig_name)
        plot.scatter(self.queries,self.results)
        plot.title(self.fig_name)
        if self.interactive: plot.show()
        else:
            plot.savefig(f'{self.path}/{self.fig_name}_{self.iteration:05d}.png')
            plot.clf()

@dataclass
class PlotAllDataPointsEvaluator(LogingEvaluator):
    """
    PlotALlDataPointsEvaluator()
    | **Description**
    |   This evaluator plots all of the experiment's data points after the experiment has concluded.

    """
    interactive: bool = False
    folder: str = "fig"
    fig_name:str = "AllData"

    data_pools: ResultDataPools = post_init()

    def register(self, experiment: Experiment):
        """
        register(self, experiment) -> None
        | **Description**
        |   Modifies the experiment to plot all data points after running the experiment.
        |   Requires the experiment's data pool to be a ResultDataPool.

        :param experiment: The experiment to be evaluated
        :type experiment: Experiment
        :raises: TypeError if self.experiment.data_pools is not a ResultDataPools
        """
        super().register(experiment)

        os.makedirs(self.path, exist_ok=True)

        if isinstance(self.experiment.data_pools, ResultDataPools):
            self.data_pools = self.experiment.data_pools
        else:
            raise TypeError(f"PlotNewDataPointsEvaluator requires ResultDataPools")
        
        self.experiment.run = Evaluate(self.experiment.run)
        self.experiment.run.post(self.log_data)
    
    def log_data(self, exp_nr):
        """
        log_data(self) -> None
        | **Description**
        |   Plots all of the experiment's data points.
        """
        queries = self.data_pools.result.queries
        results = self.data_pools.result.results

        fig = plot.figure(self.fig_name)
        plot.scatter(queries,results)
        plot.title(self.fig_name)
        if self.interactive: plot.show()
        else:
            plot.savefig(f'{self.path}/{self.fig_name}.png')
            plot.clf()

@dataclass
class PlotQueryDistEvaluator(LogingEvaluator):
    """
    PlotQueryDistEvaluator()
    | **Description**
    |   This evaluator plots all of the experiment's new queries continuously as they are made as a histogram.

    """
    interactive: bool = False
    folder: str = "fig"
    fig_name:str = "Query distribution"

    queries: NDArray[Shape["query_nr, ... query_dim"], Number] = field(init = False, default = None) # type: ignore

    def register(self, experiment: Experiment):
        """
        register(self, experiment) -> None
        | **Description**
        |   Modifies the experiment to plot the new queries as new frames of the histogram.
        |   Requires the experiment's oracles to be a POracles.

        :param experiment: The experiment to be evaluated
        :type experiment: Experiment
        :raises: TypeError if self.experiment.oracles is not a POracles
        """
        super().register(experiment)

        if isinstance(self.experiment.oracles, POracles):
            self.experiment.oracles.process.add = Evaluate(self.experiment.oracles.process.add)
            self.experiment.oracles.process.add.pre(self.plot_query_dist)
        else:
            raise TypeError(f"PlotQueryDistEvaluator requires POracles")

        self.queries: NDArray[Shape["query_nr, ... query_dim"], Number] = None # type: ignore

    def plot_query_dist(self, queries):
        """
        plot_query_dist(self, queries) -> None
        | **Description**
        |   Plots the given queries on a new frame of the histogram.

        :param queries: New queries going to the query queue
        :type queries: Tuple[NDArray[Shape["query_nr, ... query_dim"], Number]
        """
        if self.queries is None:
            self.queries = queries
        else:
            self.queries = np.concatenate((self.queries, queries))

        fig = plot.figure(self.fig_name)
        plot.hist(self.queries)
        plot.title(self.fig_name)
        if self.interactive: plot.show()
        else:
            plot.savefig(f'{self.path}/{self.fig_name}_{self.iteration:05d}.png')
            plot.clf()

class PlotSampledQueriesEvaluator(LogingEvaluator):
    """
    PlotSampledQueriesEvaluator()
    | **Description**
    |   This evaluator plots the experiment's selected queries.

    """
    interactive: bool = True
    folder: str = "fig"
    fig_name:str = "Sampled queries"

    def register(self, experiment: Experiment):
        """
        register(self, experiment) -> None
        | **Description**
        |   Modifies the experiment to plot the queries passing its selection criteria.

        :param experiment: The experiment to be evaluated
        :type experiment: Experiments
        """
        super().register(experiment)

        self.experiment.experiment_modules.query_selector.query_optimizer.selection_criteria.query = Evaluate(self.experiment.experiment_modules.query_selector.query_optimizer.selection_criteria.query)
        self.experiment.experiment_modules.query_selector.query_optimizer.selection_criteria.query.pre(self.plot_queries)

    def plot_queries(self, queries):
        """
        plot_queries(self, queries) -> None
        | **Description**
        |   Plots the given queries.

        :param queries: New queries going to the query queue
        :type queries: Tuple[NDArray[Shape["query_nr, ... query_dim"], Number]
        """
        fig = plot.figure(self.fig_name)
        plot.scatter(queries, [0 for i in range(queries.shape[0])])
        plot.title(self.fig_name)
        if self.interactive: plot.show()
        else:
            plot.savefig(f'{self.path}/{self.fig_name}_{self.iteration:05d}.png')
            plot.clf()

@dataclass
class LogOracleEvaluator(LogingEvaluator):
    """
    LogOracleEvaluator()
    | **Description**
    |   This evaluator logs all queries processed by the oracle.

    """
    folder: str = "log"
    file_name:str = "oracle_data"

    def register(self, experiment: Experiment):
        """
        register(self, experiment) -> None
        | **Description**
        |   Modifies the experiment to log all queries added to the oracle process.
        |   Requires the experiment's oracles to be a POracles.

        :param experiment: The experiment to be evaluated
        :type experiment: Experiments
        :raises: TypeError if self.experiment.oracles is not a POracles
        """
        super().register(experiment)

        if isinstance(self.experiment.oracles, POracles):
            self.experiment.oracles.process.add = Evaluate(self.experiment.oracles.process.add)
            self.experiment.oracles.process.add.pre(self.save_query)
        else:
            raise TypeError(f"LogOracleEvaluator requires POracles")

        self.experiment.run = Evaluate(self.experiment.run)
        self.experiment.run.post(self.log_data)

        self.queries = None

    def save_query(self, queries):
        """
        save_query(self, queries) -> None
        | **Description**
        |   Saves the given query with the previously saved queries.

        :param queries: New queries going to the oracle process
        :type queries: Tuple[NDArray[Shape["query_nr, ... query_dim"], Number]
        """
        if self.queries is None:
            self.queries = queries
        else:
            self.queries = np.concatenate((self.queries, queries))
    
    def log_data(self, exp_nr):
        """
        log_data(self) -> None
        | **Description**
        |   Logs all saved queries to an ```.npy``` file.
        """
        if not self.queries is None:
            np.save(f'{self.path}/{self.file_name}.npy', self.queries) 

@dataclass
class LogStreamEvaluator(LogingEvaluator):
    """
    LogStreamEvaluator()
    | **Description**
    |   This evaluator logs all data points added to the experiment's data pools stream.

    """
    folder: str = "log"
    file_name:str = "stream"

    def register(self, experiment: Experiment):
        """
        register(self, experiment) -> None
        | **Description**
        |   Modifies the experiment to log all data points added to the data pools stream.
        |   Requires the experiment's data pools to be a StreamDataPools.

        :param experiment: The experiment to be evaluated
        :type experiment: Experiments
        :raises: TypeError if self.experiment.data_pools is not a StreamDataPools
        """
        super().register(experiment)

        if isinstance(self.experiment.data_pools, StreamDataPools):
            self.experiment.data_pools.stream.add = Evaluate(self.experiment.data_pools.stream.add)
            self.experiment.data_pools.stream.add.pre(self.save_stream)
        else:
            raise TypeError(f"LogStreamEvaluator requires StreamDataPools")

        self.experiment.run = Evaluate(self.experiment.run)
        self.experiment.run.post(self.log_data)

        self.stream = None

    def save_stream(self, data):
        """
        save_stream(self, data) -> None
        | **Description**
        |   Saves the given data point with the previously saved data points.

        :param queries: New data points going to the data pools stream
        :type queries: Tuple[Tuple[NDArray[Shape["query_nr, ... query_dim"], Number], Tuple[NDArray[Shape["result_nr, ... result_dim"], Number]]
        """
        combined_data = np.concatenate((data[0], data[1]), axis=1)
        if self.stream is None:
            self.stream = combined_data
        else:
            self.stream = np.concatenate((self.stream, combined_data))

    
    def log_data(self):
        """
        log_data(self) -> None
        | **Description**
        |   Logs all saved data points to an ```.npy``` file if there is at least one data point.
        """
        if not self.stream is None:
            np.save(f'{self.path}/{self.file_name}.npy', self.stream)

@dataclass
class LogProcessEvaluator(LogingEvaluator):
    """
    LogProcessEvaluator()
    | **Description**
    |   This evaluator logs all data points added to the experiment's data pools process.

    """
    folder: str = "log"
    file_name:str = "process"

    def register(self, experiment: Experiment):
        """
        register(self, experiment) -> None
        | **Description**
        |   Modifies the experiment to log all data points added to the data pools process.
        |   Requires the experiment's data pools to be a ProcessDataPools.

        :param experiment: The experiment to be evaluated
        :type experiment: Experiments
        :raises: TypeError if self.experiment.data_pools is not a ProcessDataPools
        """
        super().register(experiment)

        if isinstance(self.experiment.data_pools, ProcessDataPools):
            self.experiment.data_pools.process.add = Evaluate(self.experiment.data_pools.process.add)
            self.experiment.data_pools.process.add.pre(self.save_process)
        else:
            raise TypeError(f"LogProcessEvaluator requires ProcessDataPools")

        self.experiment.run = Evaluate(self.experiment.run)
        self.experiment.run.post(self.log_data)

        self.process = None

    def save_process(self, data):
        """
        save_process(self, data) -> None
        | **Description**
        |   Saves the given data point with the previously saved data points.

        :param queries: New data points going to the data pools process
        :type queries: Tuple[Tuple[NDArray[Shape["query_nr, ... query_dim"], Number], Tuple[NDArray[Shape["result_nr, ... result_dim"], Number]]
        """
        combined_data = np.concatenate((data[0], data[1]), axis=1)
        if self.process is None:
            self.process = combined_data
        else:
            self.process = np.concatenate((self.process, combined_data))
    
    def log_data(self):
        """
        log_data(self) -> None
        | **Description**
        |   Logs all saved data points to an ```.npy``` file if there is at least one data point.
        """
        if not self.process is None:
            np.save(f'{self.path}/{self.file_name}.npy', self.process)

@dataclass
class LogResultEvaluator(LogingEvaluator):
    """
    LogProcessEvaluator()
    | **Description**
    |   This evaluator logs all results added to the experiment's data pools results.

    """
    folder: str = "log"
    file_name:str = "result"

    def register(self, experiment: Experiment):
        """
        register(self, experiment) -> None
        | **Description**
        |   Modifies the experiment to log all results added to the data pools results.
        |   Requires the experiment's data pools to be a ResultDataPools.

        :param experiment: The experiment to be evaluated
        :type experiment: Experiments
        :raises: TypeError if self.experiment.data_pools is not a ResultDataPools
        """
        super().register(experiment)

        if isinstance(self.experiment.data_pools, ResultDataPools):
            self.experiment.data_pools.result.add = Evaluate(self.experiment.data_pools.result.add)
            self.experiment.data_pools.result.add.pre(self.save_result)
        else:
            raise TypeError(f"LogResultEvaluator requires ResultDataPools")

        self.experiment.run = Evaluate(self.experiment.run)
        self.experiment.run.post(self.log_data)

        self.results = None

    
    def save_result(self, data):
        """
        save_result(self, data) -> None
        | **Description**
        |   Saves the given result with the previously saved results.

        :param queries: New results going to the data pools results
        :type queries: Tuple[Tuple[NDArray[Shape["query_nr, ... query_dim"], Number], Tuple[NDArray[Shape["result_nr, ... result_dim"], Number]]
        """
        combined_data = np.concatenate((data[0], data[1]), axis=1)
        if self.results is None:
            self.results = combined_data
        else:
            self.results = np.concatenate((self.results, combined_data))
    
    def log_data(self, exp_nr):
        """
        log_data(self) -> None
        | **Description**
        |   Logs all saved results to an ```.npy``` file if there is at least one result.
        """
        if not self.results is None:
            np.save(f'{self.path}/{self.file_name}.npy', self.results)

@dataclass
class LogAllEvaluator(LogingEvaluator):
    """
    LogAllEvaluator()
    | **Description**
    |   This evaluator logs combines the LogStreamEvaluator, LogProcessEvaluator and LogResultEvaluator.

    """
    folder: str = "log"
    file_name:str = "all_data"

    lsev: LogStreamEvaluator = post_init()
    lpev: LogProcessEvaluator = post_init()
    lrev: LogResultEvaluator = post_init()

    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Initializes the LogStream-, LogProcess- and LogResult- evaluators
        """
        super().post_init()
        self.lsev = LogStreamEvaluator(folder=self.folder, file_name=f"{self.file_name}_stream")()
        self.lpev = LogProcessEvaluator(folder=self.folder, file_name=f"{self.file_name}_process")()
        self.lrev = LogResultEvaluator(folder=self.folder, file_name=f"{self.file_name}_result")()

    def register(self, experiment: Experiment):
        """
        register(self, experiment) -> None
        | **Description**
        |   Registers the experiment with all 3 above listed evaluators.
        """
        super().register(experiment)
        self.lsev.register(experiment = experiment)
        self.lpev.register(experiment = experiment)
        self.lrev.register(experiment = experiment)

@dataclass
class LogTVPGTEvaluator(LogingEvaluator):
    """
    LogTVPGTEvaluator()
    | **Description**
    |   The Log Time Varying Process Ground Truth Evaluator ---

    """
    folder: str = "log"
    file_name:str = "gt_data"

    def register(self, experiment: Experiment):
        """
        register(self, experiment) -> None
        | **Description**
        |   Modifies the experiment to log all results coming form the process' update.
        |   Requires the experiment's process to be a DelayedProcess.

        :param experiment: The experiment to be evaluated
        :type experiment: Experiments
        :raises: TypeError if self.experiment.process is not a DelayedProcess
        """
        super().register(experiment)

        if isinstance(self.experiment.process, DelayedProcess):
            self.experiment.process.update = Evaluate(self.experiment.process.update)
            self.experiment.process.update.post(self.save_gt)
        else:
            raise TypeError(f"LogTVPGTEvaluator requires DelayedProcess")

        self.experiment.run = Evaluate(self.experiment.run)
        self.experiment.run.post(self.log_data)

        self.gt = None

    def save_gt(self, data):
        """
        save_gt(self, data) -> None
        | **Description**
        |   Saves the given data (ground truths) points with the previously saved data points.

        :param queries: New results going to the data pools results
        :type queries: Tuple[Tuple[NDArray[Shape["query_nr, ... query_dim"], Number], Tuple[NDArray[Shape["result_nr, ... result_dim"], Number]]
        """
        gt_queries, gt_results = data

        combined_data = np.concatenate((gt_queries, gt_results), axis=1)

        if self.gt is None:
            self.gt = combined_data
        else:
           self.gt = np.concatenate((self.gt, combined_data))
    
    def log_data(self):
        """
        log_data(self) -> None
        | **Description**
        |   Logs all saved data points (ground truths) to an ```.npy``` file if there is at least one data point.
        """
        if not self.gt is None:
            np.save(f'{self.path}/{self.file_name}.npy', self.gt)

