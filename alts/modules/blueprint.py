from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

from alts.core.blueprint import Blueprint

from alts.modules.data_process.time_source import IterationTimeSource
from alts.modules.data_process.process import DataSourceProcess
from alts.modules.oracle.query_queue import FCFSQueryQueue
from alts.modules.oracle.data_source import LineDataSource
from alts.modules.stopping_criteria import TimeStoppingCriteria
from alts.modules.queried_data_pool import FlatQueriedDataPool
from alts.modules.query.query_sampler import LatinHypercubeQuerySampler, UniformQuerySampler
from alts.core.experiment_modules import InitQueryExperimentModules
from alts.core.query.query_selector import ResultQuerySelector
from alts.modules.query.query_optimizer import NoQueryOptimizer
from alts.modules.query.query_decider import AllQueryDecider
from alts.modules.evaluator import PlotNewDataPointsEvaluator, PrintExpTimeEvaluator
from alts.core.oracle.oracles import POracles
from alts.core.data.data_pools import ResultDataPools

if TYPE_CHECKING:
    from typing import Iterable, Optional

    from alts.core.data_process.time_source import TimeSource
    from alts.core.data_process.process import Process
    from alts.core.stopping_criteria import StoppingCriteria
    from alts.core.experiment_modules import ExperimentModules
    from alts.core.evaluator import Evaluator
    from alts.core.oracle.oracles import Oracles
    from alts.core.data.data_pools import DataPools

@dataclass
class BaselineBlueprint(Blueprint):
    repeat: int = 1

    time_source: TimeSource = IterationTimeSource()

    oracles: Oracles = POracles(process = FCFSQueryQueue())

    data_pools: DataPools = ResultDataPools(result=FlatQueriedDataPool())

    process: Process = DataSourceProcess(
        data_source=LineDataSource()
    )

    stopping_criteria: StoppingCriteria = TimeStoppingCriteria(stop_time=100)


    experiment_modules: ExperimentModules = InitQueryExperimentModules(
        initial_query_sampler = LatinHypercubeQuerySampler(num_queries=10),
        query_selector=ResultQuerySelector(
            query_optimizer=NoQueryOptimizer(query_sampler=UniformQuerySampler()),
            query_decider=AllQueryDecider(),
            ),
        )

    evaluators: Iterable[Evaluator] = (PlotNewDataPointsEvaluator(), PrintExpTimeEvaluator())
