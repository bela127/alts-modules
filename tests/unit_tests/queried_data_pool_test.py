#Test version 2.0 as of 31.08.2025
import alts.modules.queried_data_pool as qdps

from alts.modules.data_process.process import DataSourceProcess
from alts.modules.oracle.data_source import RandomUniformDataSource
from alts.core.experiment_runner import ExperimentRunner
from alts.core.data.queried_data_pool import QueriedDataPool

import alts.modules.testing_modules as tm
import numpy as np
import pytest

"""
| **Test aims**
|   The queried data pool modules are tested for:
|   - Correct input and output constraints
"""

queried_data_pools = tm.query_for_members(qdps, QueriedDataPool)
special_queried_data_pools = [
    
]
simple_queried_data_pools = np.setdiff1d(queried_data_pools, special_queried_data_pools, assume_unique=True)

shape_values = [(1,), (1,1), (2,2), (5,3,2), (4,1,2,1)]

@pytest.mark.parametrize("result_shape", shape_values)
@pytest.mark.parametrize("query_shape", shape_values)
def test_FlatQueriedDataPool(query_shape: tuple, result_shape: tuple):
    bp = tm.TestBlueprint(process=DataSourceProcess(data_source=RandomUniformDataSource(query_shape=query_shape, result_shape=result_shape)),
                          evaluators=(tm.ConstraintEvaluator(func_path="data_pools.result.query", q_index=slice(None,None,None), r_index=1),))
    er = ExperimentRunner([bp])
    er.run_experiment(bp)


special_assginments = {

}


@pytest.mark.parametrize("special_queried_data_pool", special_queried_data_pools)
def test_special(special_queried_data_pool: type[QueriedDataPool]):
    if (special_queried_data_pool not in special_assginments):
        pytest.xfail(f"Special declared QueriedDataPool {special_queried_data_pool.__name__} not tested")
    elif special_assginments[special_queried_data_pool] is None:
        pytest.xfail(f"Not yet implemented: {special_queried_data_pool.__name__}")
    elif special_assginments[special_queried_data_pool] == "abstract":
        pytest.skip(f"Abstract: {special_queried_data_pool.__name__}")