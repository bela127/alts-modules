from alts.core.query.query_sampler import QuerySampler
from alts.modules.query.query_optimizer import NoQueryOptimizer
import alts.modules.query.query_sampler as qss

from alts.modules.data_process.process import DataSourceProcess
from alts.modules.oracle.data_source import RandomUniformDataSource
from alts.core.experiment_runner import ExperimentRunner
from alts.modules.query.query_decider import AllQueryDecider
from alts.modules.query.query_optimizer import NoQueryOptimizer
from alts.modules.query.query_sampler import UniformQuerySampler
from alts.core.query.query_selector import ResultQuerySelector
from alts.core.experiment_modules import InitQueryExperimentModules

import alts.modules.testing_modules as tm
import numpy as np
import pytest

"""
| **Test aims**
|   The query samplers are tested for:
|   - Correct output dimensions with respect to paramters
|   - Handling of nonsensical user parameters
"""
query_samplers = tm.query_for_members(qss, QuerySampler)
special_query_samplers = [

]
simple_query_samplers = np.setdiff1d(query_samplers, special_query_samplers, assume_unique=True)


queries = np.array([[0],[0.25],[0.5],[0.75],[1]])
shape_values = [(1,), (1,1), (2,2), (5,3,2), (2,3,4,1)]
    
@pytest.mark.parametrize("query_shape", shape_values)
@pytest.mark.parametrize("qs", simple_query_samplers)
def test_QuerySampler(qs: QuerySampler, query_shape):
    bp = tm.TestBlueprint(process=DataSourceProcess(data_source=RandomUniformDataSource(query_shape=query_shape)),
                               experiment_modules=InitQueryExperimentModules(initial_query_sampler=UniformQuerySampler(num_queries=10), query_selector=ResultQuerySelector(query_optimizer=NoQueryOptimizer(query_sampler=qs()), query_decider=AllQueryDecider())),
                               evaluators=(tm.ConstraintEvaluator(func_path="experiment_modules.query_selector.query_optimizer.query_sampler.sample", q_index=None, r_index=slice(None,None,None)),))
    er = ExperimentRunner([bp])
    er.run_experiment(bp)
    
special_assginments = {

}

@pytest.mark.parametrize("special_query_sampler", special_query_samplers)
def test_special_tested(special_query_sampler: type[QuerySampler]):
    if (special_query_sampler not in special_assginments):
        pytest.xfail(f"Special declared QuerySampler {special_query_sampler.__name__} not tested")
    if special_assginments[special_query_sampler] is None:
        pytest.xfail(f"Not yet implemented: {special_query_sampler.__name__}")