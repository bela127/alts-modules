#Test version 1.2 as of 18.07.2025
from alts.core.query.query_decider import QueryDecider
import alts.modules.query.query_decider as qdm

from alts.modules.data_process.process import DataSourceProcess
from alts.modules.oracle.data_source import LineDataSource
import alts.modules.blueprint as bps
from alts.core.experiment import Experiment
from alts.modules.query.query_optimizer import NoQueryOptimizer
from alts.modules.query.query_sampler import LatinHypercubeQuerySampler, UniformQuerySampler
from alts.core.query.query_selector import ResultQuerySelector
from alts.core.experiment_modules import InitQueryExperimentModules
from alts.core.evaluator import Evaluator

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

shape_values = [(1,), (1,1)]#, (2,2), (5,3,2), (2,3,4,1)]

k_values = [1,3,10,100]#,0]
@pytest.mark.parametrize("k", k_values)
@pytest.mark.parametrize("shape", shape_values)
def test_TopKQueryDecider(k: int, shape: tuple):
    bp = bps.BaselineBlueprint(process=DataSourceProcess(data_source=LineDataSource(query_shape=shape)),
                               experiment_modules=InitQueryExperimentModules(initial_query_sampler=LatinHypercubeQuerySampler(num_queries=10), query_selector=ResultQuerySelector(query_optimizer=NoQueryOptimizer(query_sampler=UniformQuerySampler()), query_decider=qdm.TopKQueryDecider(k))),
                               evaluators=(tm.ShapeEvaluator(shape=(k,)+shape, func="experiment_modules.query_selector.query_decider.decide", idx=(1,)),
                                           ))
    exp = Experiment(bp, 1)
    exp.run()
