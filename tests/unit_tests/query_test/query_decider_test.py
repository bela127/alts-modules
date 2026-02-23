#Test version 2.1 as of 06.09.2025
from alts.core.query.query_decider import QueryDecider
import alts.modules.query.query_decider as qdm

from alts.modules.data_process.process import DataSourceProcess
from alts.modules.oracle.data_source import RandomUniformDataSource
from alts.core.experiment_runner import ExperimentRunner
from alts.modules.query.query_optimizer import NoQueryOptimizer
from alts.modules.query.query_sampler import UniformQuerySampler
from alts.core.query.query_selector import ResultQuerySelector
from alts.core.experiment_modules import InitQueryExperimentModules

import alts.modules.testing_modules as tm
import numpy as np
import pytest

"""
| **Test aims**
|   The query deciders are tested for:
|   - Correct output dimensions with respect to paramters
|   - Handling of nonsensical user parameters
"""
query_deciders = tm.query_for_members(qdm, QueryDecider)
special_query_deciders = [
    qdm.ThresholdQueryDecider,
    qdm.TopKQueryDecider
]
simple_query_deciders = np.setdiff1d(query_deciders, special_query_deciders, assume_unique=True)

shape_values = [(1,), (1,1), (2,2), (5,3,2), (2,3,4,1)]

k_values = [1,3,10,100]
@pytest.mark.parametrize("k", k_values)
@pytest.mark.parametrize("query_shape", shape_values)
def test_topK(k: int, query_shape: tuple):
    bp = tm.TestBlueprint(process=DataSourceProcess(data_source=RandomUniformDataSource(query_shape=query_shape)),
                               experiment_modules=InitQueryExperimentModules(initial_query_sampler=UniformQuerySampler(num_queries=10), query_selector=ResultQuerySelector(query_optimizer=NoQueryOptimizer(query_sampler=UniformQuerySampler()), query_decider=qdm.TopKQueryDecider(k))),
                               evaluators=(tm.ConstraintEvaluator(func_path="experiment_modules.query_selector.query_decider.decide", q_index=slice(None,None,None), r_index=1),))
    er = ExperimentRunner([bp])
    er.run_experiment(bp)

t_values = [-0.5,0,0.5,1,10]
@pytest.mark.parametrize("t", t_values)
@pytest.mark.parametrize("query_shape", shape_values)
def test_threshold(t: int, query_shape: tuple):
    bp = tm.TestBlueprint(process=DataSourceProcess(data_source=RandomUniformDataSource(query_shape=query_shape)),
                               experiment_modules=InitQueryExperimentModules(initial_query_sampler=UniformQuerySampler(num_queries=10), query_selector=ResultQuerySelector(query_optimizer=NoQueryOptimizer(query_sampler=UniformQuerySampler()), query_decider=qdm.ThresholdQueryDecider(t))),
                               evaluators=(tm.ConstraintEvaluator(func_path="experiment_modules.query_selector.query_decider.decide", q_index=slice(None,None,None), r_index=1),))
    er = ExperimentRunner([bp])
    er.run_experiment(bp)


@pytest.mark.parametrize("query_shape", shape_values)
@pytest.mark.parametrize("qd", simple_query_deciders)
def test_basic(qd: type[QueryDecider], query_shape: tuple):
    bp = tm.TestBlueprint(process=DataSourceProcess(data_source=RandomUniformDataSource(query_shape=query_shape)),
                               experiment_modules=InitQueryExperimentModules(initial_query_sampler=UniformQuerySampler(num_queries=10), query_selector=ResultQuerySelector(query_optimizer=NoQueryOptimizer(query_sampler=UniformQuerySampler()), query_decider=qd())),
                               evaluators=(tm.ConstraintEvaluator(func_path="experiment_modules.query_selector.query_decider.decide", q_index=slice(None,None,None), r_index=1),))
    er = ExperimentRunner([bp])
    er.run_experiment(bp)


special_assginments = {
    qdm.ThresholdQueryDecider: test_threshold,
    qdm.TopKQueryDecider: test_topK
}

@pytest.mark.parametrize("special_query_decider", special_query_deciders)
def test_special_tested(special_query_decider: type[QueryDecider]):
    assert special_query_decider in special_assginments, f"Special declared QueryDecider {special_query_decider.__name__} not tested"
    if special_assginments[special_query_decider] is None:
        pytest.skip(f"Not yet implemented: {special_query_decider.__name__}")