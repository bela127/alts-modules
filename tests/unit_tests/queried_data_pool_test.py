#Test version 2.0 as of 31.08.2025
import alts.modules.queried_data_pool as qdps

from alts.modules.data_process.process import DataSourceProcess
from alts.modules.oracle.data_source import RandomUniformDataSource
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

shape_values = [(1,), (1,1), (2,2), (5,3,2), (4,1,2,1)]

@pytest.mark.parametrize("result_shape", shape_values)
@pytest.mark.parametrize("query_shape", shape_values)
def test_FlatQueriedDataPool(query_shape: tuple, result_shape: tuple):
    bp = tm.TestBlueprint(process=DataSourceProcess(data_source=RandomUniformDataSource(query_shape=query_shape, result_shape=result_shape)),
                          evaluators=(tm.ConstrainEvaluator(func_path="data_pools.result.add", q_index=0),
                                      tm.ConstrainEvaluator(func_path="data_pools.result.query", q_index=slice(None,None,None), r_index=1)))
    er = ExperimentRunner([bp])
    er.run_experiment(bp)
    