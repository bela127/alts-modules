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
|   The data sampler modules are tested for:
|   - Results inside constraints if queries inside constraints
"""

data_samplers = [
    dss.KDTreeKNNDataSampler,
    dss.KDTreeRegionDataSampler
]

shape_values = [(1,), (1,1), (2,2), (5,3,2), (2,3,4,1)]

@pytest.mark.parametrize("query_shape", shape_values)
def test_KDTreeKNNDataSampler(query_shape: tuple):
    bp = tm.TestBlueprint(process=DataSourceProcess(data_source=RandomUniformDataSource(query_shape=query_shape)),
                               evaluators=(tm.ConstraintEvaluator(func_path="oracles.process.pop", r_index=slice(None,None,None)),))
    er = ExperimentRunner([bp])
    er.run_experiment(bp)