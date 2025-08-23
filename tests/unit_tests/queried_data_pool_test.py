from alts.core.data.queried_data_pool import QueriedDataPool
import alts.modules.queried_data_pool as qdps
from alts.core.data.constrains import QueryConstrain, ResultConstrain

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
|   The queried data pool modules are tested for:
|   - Correct output dimensions
"""

queried_data_pools = [
    qdps.FlatQueriedDataPool
]

def add_warp(self, func, data_points):
    queries, results = data_points
    print(self.queries)
    print(queries)
    self.queries = np.concatenate((self.queries, queries))
    self.results = np.concatenate((self.results, results))
    
    self.last_queries = queries
    self.last_results = results

    self.request_update()

shape_values = [(1,), (1,1)]#, (2,2), (5,3,2), (2,3,4,1)]

@pytest.mark.parametrize("result_shape", shape_values)
@pytest.mark.parametrize("query_shape", shape_values)
def test_FlatQueriedDataPool(query_shape: tuple, result_shape: tuple):
    bp = bps.BaselineBlueprint(process=DataSourceProcess(data_source=LineDataSource(query_shape=query_shape, result_shape=result_shape)),
                               evaluators=tm.ACEEvaluator(func="data_pools.super(FlatQueriedDataPool,result).add",warp=add_warp))
    exp = Experiment(bp, 1)
    exp.run()