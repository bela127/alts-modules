from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field
import os
import time


from alts.core.evaluator import Evaluator, Evaluate, LogingEvaluator
from alts.modules.data_process.process import DataSourceProcess

import numpy as np
from matplotlib import pyplot as plot # type: ignore


if TYPE_CHECKING:
    from alts.core.experiment import Experiment
    from nptyping import  NDArray, Number, Shape
    from alts.core.oracle.data_source import DataSource


class PrintNewDataPointsEvaluator(Evaluator):

    def register(self, experiment: Experiment):
        super().register(experiment)

        self.experiment.observable_results.add = Evaluate(self.experiment.observable_results.add)
        self.experiment.observable_results.add.pre(self.log_new_data_points)

    def log_new_data_points(self, data_points):
        print(data_points)

class PrintQueryEvaluator(Evaluator):

    def register(self, experiment: Experiment):
        super().register(experiment)

        self.experiment.oracle.request = Evaluate(self.experiment.oracle.request)
        self.experiment.oracle.request.pre(self.print_query)

    def print_query(self, query):
        print("Queried: \n",query)
class PrintExpTimeEvaluator(Evaluator):

    def register(self, experiment: Experiment):
        super().register(experiment)

        self.experiment.run = Evaluate(self.experiment.run)
        self.experiment.run.pre(self.start_time)
        self.experiment.run.post(self.end_time)

    def start_time(self):
        print("Start timing")
        self.start = time.time()
    
    def end_time(self, exp_nr):
        end = time.time()
        print("Time: ",end - self.start)

class LogNewDataPointsEvaluator(LogingEvaluator):
    def __init__(self, logger) -> None:
        super().__init__()
        self.logger = logger

    def register(self, experiment: Experiment):
        super().register(experiment)

        self.experiment.observable_results.add = Evaluate(self.experiment.observable_results.add)
        self.experiment.observable_results.add.pre(self.log_new_data_points)

    def log_new_data_points(self, data_points):
        # self.logger(data_points)
        ...

@dataclass
class PlotNewDataPointsEvaluator(LogingEvaluator):
    interactive: bool = False
    folder: str = "fig"
    fig_name:str = "Data"

    queries: NDArray[Number, Shape["query_nr, ... query_dim"]] = field(init = False, default = None)
    results: NDArray[Number, Shape["query_nr, ... result_dim"]] = field(init = False, default = None)
    iteration: int = field(init = False, default = 0)

    def register(self, experiment: Experiment):
        super().register(experiment)

        os.makedirs(self.path, exist_ok=True)

        self.experiment.result_data_pool.add = Evaluate(self.experiment.result_data_pool.add)
        self.experiment.result_data_pool.add.pre(self.plot_new_data_points)

        self.queries: NDArray[Number, Shape["query_nr, ... query_dim"]] = None
        self.results: NDArray[Number, Shape["query_nr, ... result_dim"]] = None
        self.iteration = 0

    def plot_new_data_points(self, data_points):

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

        self.iteration += 1

@dataclass
class PlotQueryDistEvaluator(LogingEvaluator):
    interactive: bool = False
    folder: str = "fig"
    fig_name:str = "Query distribution"

    queries: NDArray[Number, Shape["query_nr, ... query_dim"]] = field(init = False, default = None)
    iteration: int = field(init = False, default = 0)

    def register(self, experiment: Experiment):
        super().register(experiment)

        self.experiment.oracle.request = Evaluate(self.experiment.oracle.request)
        self.experiment.oracle.request.pre(self.plot_query_dist)

        self.queries: NDArray[Number, Shape["query_nr, ... query_dim"]] = None

    def plot_query_dist(self, query_candidate):

        if self.queries is None:
            self.queries = query_candidate
        else:
            self.queries = np.concatenate((self.queries, query_candidate))

        fig = plot.figure(self.fig_name)
        plot.hist(self.queries)
        plot.title(self.fig_name)
        if self.interactive: plot.show()
        else:
            plot.savefig(f'{self.path}/{self.fig_name}_{self.iteration:05d}.png')
            plot.clf()

        self.iteration += 1

class PlotSampledQueriesEvaluator(LogingEvaluator):
    interactive: bool = True
    folder: str = "fig"
    fig_name:str = "Sampled queries"

    iteration: int = field(init = False, default = 0)

    def register(self, experiment: Experiment):
        super().register(experiment)

        self.experiment.experiment_modules.query_selector.query_optimizer.selection_criteria.query = Evaluate(self.experiment.experiment_modules.query_selector.query_optimizer.selection_criteria.query)
        self.experiment.experiment_modules.query_selector.query_optimizer.selection_criteria.query.pre(self.plot_queries)

    def plot_queries(self, queries):

        fig = plot.figure(self.fig_name)
        plot.scatter(queries, [0 for i in range(queries.shape[0])])
        plot.title(self.fig_name)
        if self.interactive: plot.show()
        else:
            plot.savefig(f'{self.path}/{self.fig_name}_{self.iteration:05d}.png')
            plot.clf()

        self.iteration += 1


@dataclass
class LogOracleEvaluator(LogingEvaluator):
    folder: str = "log"
    file_name:str = "oracle_data"

    def register(self, experiment: Experiment):
        super().register(experiment)

        self.experiment.oracle.request = Evaluate(self.experiment.oracle.request)
        self.experiment.oracle.request.pre(self.save_query)

        self.experiment.run = Evaluate(self.experiment.run)
        self.experiment.run.post(self.log_data)

        self.queries = None

    def save_query(self, queries):
        if self.queries is None:
            self.queries = queries
        else:
            self.queries = np.concatenate((self.queries, queries))
    
    def log_data(self, exp_nr):

        np.save(f'{self.path}/{self.file_name}.npy', self.queries)


@dataclass
class LogAllEvaluator(LogingEvaluator):
    folder: str = "log"
    file_name:str = "all_data"

    def register(self, experiment: Experiment):
        super().register(experiment)

        self.experiment.stream_data_pool.add = Evaluate(self.experiment.stream_data_pool.add)
        self.experiment.stream_data_pool.add.pre(self.save_stream)

        self.experiment.process_data_pool.add = Evaluate(self.experiment.process_data_pool.add)
        self.experiment.process_data_pool.add.pre(self.save_process)

        self.experiment.result_data_pool.add = Evaluate(self.experiment.result_data_pool.add)
        self.experiment.result_data_pool.add.pre(self.save_result)

        self.experiment.run = Evaluate(self.experiment.run)
        self.experiment.run.post(self.log_data)

        self.stream = None
        self.process = None
        self.results = None

    def save_stream(self, data):
        combined_data = np.concatenate((data[0], data[1]), axis=1)
        if self.stream is None:
            self.stream = combined_data
        else:
            self.stream = np.concatenate((self.stream, combined_data))

    def save_process(self, data):
        combined_data = np.concatenate((data[0], data[1]), axis=1)
        if self.process is None:
            self.process = combined_data
        else:
            self.process = np.concatenate((self.process, combined_data))
    
    def save_result(self, data):
        combined_data = np.concatenate((data[0], data[1]), axis=1)
        if self.results is None:
            self.results = combined_data
        else:
            self.results = np.concatenate((self.results, combined_data))
    
    def log_data(self, exp_nr):
        np.save(f'{self.path}/{self.file_name}_stream.npy', self.stream)
        np.save(f'{self.path}/{self.file_name}_process.npy', self.process)
        np.save(f'{self.path}/{self.file_name}_result.npy', self.results)

@dataclass
class LogTVPGTEvaluator(LogingEvaluator):
    folder: str = "log"
    file_name:str = "gt_data"

    def register(self, experiment: Experiment):
        super().register(experiment)

        if not isinstance(self.experiment.process, DataSourceProcess):
            raise ValueError("for this evaluator the Process needs to be a DataSourceProcess")
        self.ds = self.experiment.process.data_source

        self.experiment.time_behavior.query = Evaluate(self.experiment.time_behavior.query)
        self.experiment.time_behavior.query.post(self.save_gt)

        self.experiment.run = Evaluate(self.experiment.run)
        self.experiment.run.post(self.log_data)

        self.gt = None

    def save_gt(self, data):
        times, vars = data

        queries = self.experiment.oracle.query_queue.last

        query = np.concatenate((times, queries, vars), axis=1)
        data = self.ds.query(query)

        combined_data = np.concatenate((data[0], data[1]), axis=1)

        if self.gt is None:
            self.gt = combined_data
        else:
           self.gt = np.concatenate((self.gt, combined_data))
    
    def log_data(self, exp_nr):
        np.save(f'{self.path}/{self.file_name}.npy', self.gt)

