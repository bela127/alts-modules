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

    def register(self, experiment: Experiment):
        super().register(experiment)

        if isinstance(self.experiment.data_pools, ResultDataPools):
            self.experiment.data_pools.result.add = Evaluate(self.experiment.data_pools.result.add)
            self.experiment.data_pools.result.add.pre(self.print_new_data_points)
        else:
            raise TypeError(f"PrintNewDataPointsEvaluator requires ResultDataPools")

    def print_new_data_points(self, data_points):
        print(data_points)

class PrintQueryEvaluator(Evaluator):

    def register(self, experiment: Experiment):
        super().register(experiment)

        if isinstance(self.experiment.oracles, POracles):
            self.experiment.oracles.process.add = Evaluate(self.experiment.oracles.process.add)
            self.experiment.oracles.process.add.pre(self.print_query)
        else:
            raise TypeError(f"PrintQueryEvaluator requires POracles")

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

class PrintTimeSourceEvaluator(Evaluator):

    def register(self, experiment: Experiment):
        super().register(experiment)

        self.experiment.time_source.step = Evaluate(self.experiment.time_source.step)
        self.experiment.time_source.step.post(self.end_time)
    
    def end_time(self, time):
        print("Sim Unit Time: ", time)


@dataclass
class PlotNewDataPointsEvaluator(LogingEvaluator):
    interactive: bool = False
    folder: str = "fig"
    fig_name:str = "Data"

    queries: NDArray[Shape["query_nr, ... query_dim"], Number] = pre_init(None)
    results: NDArray[Shape["query_nr, ... result_dim"], Number] = pre_init(None)

    def register(self, experiment: Experiment):
        super().register(experiment)

        os.makedirs(self.path, exist_ok=True)

        if isinstance(self.experiment.data_pools, ResultDataPools):
            self.experiment.data_pools.result.add = Evaluate(self.experiment.data_pools.result.add)
            self.experiment.data_pools.result.add.pre(self.plot_new_data_points)
        else:
            raise TypeError(f"PlotNewDataPointsEvaluator requires ResultDataPools")

        self.queries: NDArray[Shape["query_nr, ... query_dim"], Number] = None
        self.results: NDArray[Shape["query_nr, ... result_dim"], Number] = None

    def plot_new_data_points(self, data_points):
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
    interactive: bool = False
    folder: str = "fig"
    fig_name:str = "AllData"

    data_pools: ResultDataPools = post_init()

    def register(self, experiment: Experiment):
        super().register(experiment)

        os.makedirs(self.path, exist_ok=True)

        if isinstance(self.experiment.data_pools, ResultDataPools):
            self.data_pools = self.experiment.data_pools
        else:
            raise TypeError(f"PlotNewDataPointsEvaluator requires ResultDataPools")
        
        self.experiment.run = Evaluate(self.experiment.run)
        self.experiment.run.post(self.log_data)
    
    def log_data(self, exp_nr):
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
    interactive: bool = False
    folder: str = "fig"
    fig_name:str = "Query distribution"

    queries: NDArray[Shape["query_nr, ... query_dim"], Number] = field(init = False, default = None)

    def register(self, experiment: Experiment):
        super().register(experiment)

        if isinstance(self.experiment.oracles, POracles):
            self.experiment.oracles.process.add = Evaluate(self.experiment.oracles.process.add)
            self.experiment.oracles.process.add.pre(self.plot_query_dist)
        else:
            raise TypeError(f"PlotQueryDistEvaluator requires POracles")

        self.queries: NDArray[Shape["query_nr, ... query_dim"], Number] = None

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

class PlotSampledQueriesEvaluator(LogingEvaluator):
    interactive: bool = True
    folder: str = "fig"
    fig_name:str = "Sampled queries"

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


@dataclass
class LogOracleEvaluator(LogingEvaluator):
    folder: str = "log"
    file_name:str = "oracle_data"

    def register(self, experiment: Experiment):
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
        if self.queries is None:
            self.queries = queries
        else:
            self.queries = np.concatenate((self.queries, queries))
    
    def log_data(self, exp_nr):

        np.save(f'{self.path}/{self.file_name}.npy', self.queries)

@dataclass
class LogStreamEvaluator(LogingEvaluator):
    folder: str = "log"
    file_name:str = "stream"

    def register(self, experiment: Experiment):
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
        combined_data = np.concatenate((data[0], data[1]), axis=1)
        if self.stream is None:
            self.stream = combined_data
        else:
            self.stream = np.concatenate((self.stream, combined_data))

    
    def log_data(self, exp_nr):
        np.save(f'{self.path}/{self.file_name}.npy', self.stream)


@dataclass
class LogProcessEvaluator(LogingEvaluator):
    folder: str = "log"
    file_name:str = "process"

    def register(self, experiment: Experiment):
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
        combined_data = np.concatenate((data[0], data[1]), axis=1)
        if self.process is None:
            self.process = combined_data
        else:
            self.process = np.concatenate((self.process, combined_data))
    
    def log_data(self, exp_nr):
        np.save(f'{self.path}/{self.file_name}.npy', self.process)

@dataclass
class LogResultEvaluator(LogingEvaluator):
    folder: str = "log"
    file_name:str = "result"

    def register(self, experiment: Experiment):
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
        combined_data = np.concatenate((data[0], data[1]), axis=1)
        if self.results is None:
            self.results = combined_data
        else:
            self.results = np.concatenate((self.results, combined_data))
    
    def log_data(self, exp_nr):
        np.save(f'{self.path}/{self.file_name}.npy', self.results)

@dataclass
class LogAllEvaluator(LogingEvaluator):
    folder: str = "log"
    file_name:str = "all_data"

    lsev: LogStreamEvaluator = post_init()
    lpev: LogProcessEvaluator = post_init()
    lrev: LogResultEvaluator = post_init()

    def post_init(self):
        super().post_init()
        self.lsev = LogStreamEvaluator(folder=self.folder, file_name=f"{self.file_name}_stream")()
        self.lpev = LogProcessEvaluator(folder=self.folder, file_name=f"{self.file_name}_process")()
        self.lrev = LogResultEvaluator(folder=self.folder, file_name=f"{self.file_name}_result")()

    def register(self, experiment: Experiment):
        super().register(experiment)
        self.lsev.register(experiment = experiment)
        self.lpev.register(experiment = experiment)
        self.lrev.register(experiment = experiment)

@dataclass
class LogTVPGTEvaluator(LogingEvaluator):
    folder: str = "log"
    file_name:str = "gt_data"

    def register(self, experiment: Experiment):
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
        gt_queries, gt_results = data

        combined_data = np.concatenate((gt_queries, gt_results), axis=1)

        if self.gt is None:
            self.gt = combined_data
        else:
           self.gt = np.concatenate((self.gt, combined_data))
    
    def log_data(self, exp_nr):
        np.save(f'{self.path}/{self.file_name}.npy', self.gt)

