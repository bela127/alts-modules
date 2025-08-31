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

shape_values = [(1,), (1,1), (2,1)]#, (5,1,2,1), (4,3,2,1,1)]

@pytest.mark.parametrize("result_shape", shape_values)
@pytest.mark.parametrize("query_shape", shape_values)
def test_FlatQueriedDataPool(query_shape: tuple, result_shape: tuple):
    
    
    bp = tm.TestBlueprint(process=DataSourceProcess(data_source=LineDataSource(query_shape=query_shape, result_shape=result_shape)),
                               evaluators=(tm.ACEEvaluator(func_path="data_pools.result.add", wrap=add_wrap),))
    er = ExperimentRunner([bp])
    if len(result_shape) >= 2 and query_shape[-1] != result_shape[-2]:
        with pytest.raises(ValueError, match="Incompatible shapes"):
            er.run_experiment(bp)
    else:
        er.run_experiment(bp)
    