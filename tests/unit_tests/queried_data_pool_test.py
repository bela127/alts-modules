import alts.modules.queried_data_pool as qdps

from alts.modules.data_process.process import DataSourceProcess
from alts.modules.oracle.data_source import LineDataSource
import alts.modules.blueprint as bps
from alts.core.experiment_runner import ExperimentRunner


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

def add_wrap(func, data_points):
    print("LOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOo")
    queries, results = data_points
    print(queries)
    print(queries.shape)
    print(results.shape)
    print("HEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
    func(data_points)
    #self.queries = np.concatenate((self.queries, queries))
    #self.results = np.concatenate((self.results, results))


shape_values = [(1,1)]#, (1,)]#, (2,2), (5,3,2), (2,3,4,1)]

@pytest.mark.parametrize("result_shape", shape_values)
@pytest.mark.parametrize("query_shape", shape_values)
def test_FlatQueriedDataPool(query_shape: tuple, result_shape: tuple):
    pytest.xfail("External Issues")
    bp = bps.BaselineBlueprint(process=DataSourceProcess(data_source=LineDataSource(query_shape=query_shape, result_shape=result_shape)),
                               evaluators=(tm.ACEEvaluator(func_path="data_pools.result.add", wrap=add_wrap),
                                           tm.ACEEvaluator(func_path="process.process_query", wrap=process_query_wrap)))
    er = ExperimentRunner([bp])
    er.run_experiment(bp)