#Test version 1.2 as of 18.07.2025
from alts.core.query.query_decider import QueryDecider
import alts.modules.query.query_decider as qdm

import alts.modules.blueprint as bps
from alts.modules.query.query_optimizer import NoQueryOptimizer
from alts.modules.query.query_sampler import LatinHypercubeQuerySampler, UniformQuerySampler
from alts.core.query.query_selector import ResultQuerySelector
from alts.core.experiment_modules import InitQueryExperimentModules

import alts.modules.tests.testing_modules as tm
import numpy as np
import pytest

"""
| **Test aims**
|   The query deciders are tested for:
|   - Correct output dimensions with respect to parameters
"""

#List of query deciders
query_deciders = [
    qdm.AllQueryDecider,
    qdm.NoQueryDecider,
    qdm.ThresholdQueryDecider,
    qdm.TopKQueryDecider
]

k_values = [0,1,3,10,100]
@pytest.mark.parametrize("k",k_values)
def test_TopKQueryDecider(k: int):
    bp = bps.BaselineBlueprint(experiment_modules=InitQueryExperimentModules(initial_query_sampler=LatinHypercubeQuerySampler(num_queries=10), query_selector=ResultQuerySelector(query_optimizer=NoQueryOptimizer(query_sampler=UniformQuerySampler()), query_decider=qdm.TopKQueryDecider(k))),
                               evaluators=)
