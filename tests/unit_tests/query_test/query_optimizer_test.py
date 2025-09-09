from alts.core.query.query_optimizer import QueryOptimizer
import alts.modules.query.query_optimizer as qos

from alts.core.experiment_modules import InitQueryExperimentModules
from alts.modules.query.query_sampler import UniformQuerySampler
from alts.modules.query.query_decider import AllQueryDecider
from alts.modules.query.query_optimizer import NoQueryOptimizer
from alts.core.query.query_selector import ResultQuerySelector

from alts.modules.data_process.process import DataSourceProcess
from alts.modules.oracle.data_source import RandomUniformDataSource
from alts.core.experiment_runner import ExperimentRunner

import alts.modules.testing_modules as tm
import numpy as np
import pytest

"""
| **Test aims**
|   The query optimizers modules are tested for:
|   - Nothing
"""

query_optimizers = [
    qos.NoQueryOptimizer,
    qos.GAQueryOptimizer,
    qos.MaxMCQueryOptimizer,
    qos.ProbWeightedMCQueryOptimizer
]

general_optimizers = [
    qos.NoQueryOptimizer,
    qos.GAQueryOptimizer,
    qos.MaxMCQueryOptimizer
]

shape_values = [(1,)]#, (1,1), (2,2), (5,3,2), (2,3,4,1)]

@pytest.mark.parametrize("query_shape", shape_values)
@pytest.mark.parametrize("qo", general_optimizers)
def test_general(qo: type[QueryOptimizer], query_shape: tuple):
    bp = tm.TestBlueprint(process=DataSourceProcess(data_source=RandomUniformDataSource(query_shape=query_shape)),
                          experiment_modules=InitQueryExperimentModules(initial_query_sampler=UniformQuerySampler(num_queries=10), query_selector=ResultQuerySelector(query_optimizer=qo(query_sampler=UniformQuerySampler()), query_decider=AllQueryDecider())),
                          evaluators=(tm.ConstrainEvaluator(func_path="experiment_modules.query_selector.query_optimizer.select", r_index=0),))
    er = ExperimentRunner([bp])
    er.run_experiment(bp)