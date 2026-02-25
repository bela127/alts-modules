from alts.core.data.data_sampler import DataSampler
import alts.modules.data_sampler as dss

from alts.modules.data_process.process import DataSourceProcess
from alts.modules.oracle.data_source import RandomUniformDataSource
from alts.core.experiment_runner import ExperimentRunner

import alts.modules.testing_modules as tm
import numpy as np
import pytest

"""
| **Test aims**
|   The query samplers are tested for:
|   - Correct output dimensions with respect to paramters
|   - Handling of nonsensical user parameters
"""
data_samplers = tm.query_for_members(dss, DataSampler)
special_data_samplers = [
    dss.KDTreeKNNDataSampler,
    dss.KDTreeRegionDataSampler
]

simple_query_samplers = np.setdiff1d(data_samplers, special_data_samplers, assume_unique=True)

shape_values = [(1,), (1,1), (2,2), (5,3,2), (2,3,4,1)]

@pytest.mark.parametrize("query_shape", shape_values)
@pytest.mark.parametrize("data_sampler", simple_query_samplers)
def test_DataSampler(data_sampler: type[DataSampler], query_shape: tuple):
    pytest.xfail("WIP") #DataSampler not yet used in test
    bp = tm.TestBlueprint(process=DataSourceProcess(data_source=RandomUniformDataSource(query_shape=query_shape)),
                               evaluators=(tm.ConstraintEvaluator(func_path="oracles.process.pop", r_index=slice(None,None,None)),))
    er = ExperimentRunner([bp])
    er.run_experiment(bp)

special_assginments = {
    
}

@pytest.mark.parametrize("special_data_sampler", special_data_samplers)
def test_special_tested(special_data_sampler: type[DataSampler]):
    if (special_data_sampler not in special_assginments):
        pytest.xfail(f"Special declared QuerySampler {special_data_sampler.__name__} not tested")
    elif special_assginments[special_data_sampler] == "abstract":
        pytest.skip(f"Abstract: {special_data_sampler.__name__}")
    elif special_assginments[special_data_sampler] is None:
        pytest.xfail(f"Not yet implemented: {special_data_sampler.__name__}")