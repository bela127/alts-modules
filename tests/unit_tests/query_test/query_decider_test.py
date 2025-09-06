#Test version 1.2 as of 18.07.2025
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

#List of query deciders
query_deciders = [
    qdm.AllQueryDecider,
    qdm.NoQueryDecider,
    qdm.ThresholdQueryDecider,
    qdm.TopKQueryDecider
]

shape_values = [(1,), (1,1), (2,2)]#, (5,3,2), (2,3,4,1)]

k_values = [1,3,10,100,0]
@pytest.mark.parametrize("k", k_values)
@pytest.mark.parametrize("query_shape", shape_values)
def test_TopKQueryDecider(k: int, query_shape: tuple):
    if k == 0: pytest.xfail("Edge case k=0")
    bp = tm.TestBlueprint(process=DataSourceProcess(data_source=RandomUniformDataSource(query_shape=query_shape)),
                               experiment_modules=InitQueryExperimentModules(initial_query_sampler=UniformQuerySampler(num_queries=10), query_selector=ResultQuerySelector(query_optimizer=NoQueryOptimizer(query_sampler=UniformQuerySampler()), query_decider=qdm.TopKQueryDecider(k))),
                               evaluators=(tm.ConstrainEvaluator(func_path="experiment_modules.query_selector.query_decider.decide", q_index=slice(None,None,None), r_index=1),))
    er = ExperimentRunner([bp])
    er.run_experiment(bp)
