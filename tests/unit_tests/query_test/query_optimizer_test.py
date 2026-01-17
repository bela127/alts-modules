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
|   - Correct output dimensions with respect to paramters
"""

query_optimizers = tm.query_for_members(qos, QueryOptimizer)
special_query_optimizers = [
    qos.ProbWeightedMCQueryOptimizer,
    qos.NoQueryOptimizer
]
simple_query_optimizers = np.setdiff1d(query_optimizers, special_query_optimizers, assume_unique=True) # type: ignore

shape_values = [(1,)]#, (1,1)]#, (2,2), (5,3,2), (2,3,4,1)]

@pytest.mark.parametrize("query_shape", shape_values)
@pytest.mark.parametrize("qo", simple_query_optimizers)
def test_basic(qo: type[QueryOptimizer], query_shape: tuple):
    bp = tm.TestBlueprint(process=DataSourceProcess(data_source=RandomUniformDataSource(query_shape=query_shape)),
                          experiment_modules=InitQueryExperimentModules(initial_query_sampler=UniformQuerySampler(num_queries=10), query_selector=ResultQuerySelector(query_optimizer=qo(), query_decider=AllQueryDecider())),
                          evaluators=(tm.ConstrainEvaluator(func_path="experiment_modules.query_selector.query_optimizer.select", r_index=0),))
    er = ExperimentRunner([bp])
    er.run_experiment(bp)

special_assginments = {
    qos.ProbWeightedMCQueryOptimizer: None,
    qos.NoQueryOptimizer: "test_NoQueryOptimizer"
}

@pytest.mark.parametrize("query_shape", shape_values)
def test_NoQueryOptimizer(query_shape: tuple):
    bp = tm.TestBlueprint(process=DataSourceProcess(data_source=RandomUniformDataSource(query_shape=query_shape)),
                          experiment_modules=InitQueryExperimentModules(initial_query_sampler=UniformQuerySampler(num_queries=10), query_selector=ResultQuerySelector(query_optimizer=qos.NoQueryOptimizer(query_sampler=UniformQuerySampler()), query_decider=AllQueryDecider())),
                          evaluators=(tm.ConstrainEvaluator(func_path="experiment_modules.query_selector.query_optimizer.select", r_index=0),))
    er = ExperimentRunner([bp])
    er.run_experiment(bp)

@pytest.mark.parametrize("special_query_optimizer", special_query_optimizers)
def test_special(special_query_optimizer: type[QueryOptimizer]):
    assert special_query_optimizer in special_assginments, f"Special declared QueryOptimizer {special_query_optimizer.__name__} not tested"
    if special_assginments[special_query_optimizer] is None:
        pytest.skip(f"Not yet implemented: {special_query_optimizer.__name__}")