from alts.core.query.query_sampler import QuerySampler
from alts.modules.query.query_optimizer import NoQueryOptimizer
import alts.modules.query.query_sampler as qss

from alts.core.evaluator import Evaluator, Evaluate
from alts.core.experiment import Experiment
from alts.modules.blueprint import BaselineBlueprint
from alts.core.experiment_modules import InitQueryExperimentModules
from alts.modules.query.query_sampler import LatinHypercubeQuerySampler, UniformQuerySampler
from alts.core.query.query_selector import ResultQuerySelector
from alts.modules.query.query_decider import AllQueryDecider
from alts.modules.query.selection_criteria import AllSelectionCriteria
from alts.modules.oracle.data_source import LineDataSource

import numpy as np
import pytest

"""
| **Test aims**
|   The query samplers modules are tested for:
|   - Correct shape of sampled data
"""

query_samplers = [
    qss.OptimalQuerySampler,
    qss.FixedQuerySampler,
    qss.FixedQuerySampler,
    qss.UniformQuerySampler,
    qss.LatinHypercubeQuerySampler,
    qss.RandomChoiceQuerySampler,
    qss.ProcessQuerySampler,
    qss.LastProcessQuerySampler,
    qss.ProcessQueueQuerySampler,
    qss.DataPoolQuerySampler,
    qss.AllDataPoolQuerySampler,
    qss.AllResultPoolQuerySampler,
    qss.AllStreamPoolQuerySampler,
    qss.AllProcessPoolQuerySampler,
    qss.LastDataPoolQuerySampler,
    qss.LastResultPoolQuerySampler,
    qss.LastStreamPoolQuerySampler,
    qss.LastProcessPoolQuerySampler
]

queries = np.array([[0],[0.25],[0.5],[0.75],[1]])
    
@pytest.mark.parametrize("qs", query_samplers)
def test_shape(qs: QuerySampler):
    exp = Experiment(BaselineBlueprint(InitQueryExperimentModules(
                               initial_query_sampler = LatinHypercubeQuerySampler(num_queries=10),
                               query_selector=ResultQuerySelector(
                                   query_optimizer=NoQueryOptimizer(query_sampler=qs()),
                                   query_decider=AllQueryDecider(),
                               )
                           )),exp_nr=1) 