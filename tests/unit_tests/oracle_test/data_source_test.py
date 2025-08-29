import alts.modules.oracle.data_source as dsm

from alts.modules.data_process.process import DataSourceProcess
from alts.modules.oracle.data_source import LineDataSource
import alts.modules.blueprint as bps
from alts.core.experiment_runner import ExperimentRunner

import alts.modules.testing_modules as tm
import pytest
import numpy as np

"""
| **Test aims**
|   The data source modules are tested for:
|   - the values 0,0.1,0.25,0.5,0.75 (for random data sources a range of results is accepted)
|   - single 1D queries
|   - multiple 1D queries
|   - multiple 2D queries
|   - 1 normal case and all edge cases for parameters
"""

#List of DataSources
data_sources = [
    dsm.BrownianDriftDataSource,
    dsm.BrownianProcessDataSource,
    dsm.CrossDataSource,
    dsm.DoubleLinearDataSource,
    dsm.ExpDataSource,#5
    dsm.GaussianProcessDataSource,
    dsm.HourglassDataSource,
    dsm.HypercubeDataSource,
    dsm.HyperSphereDataSource,
    dsm.IndependentDataSource,#10
    dsm.InterpolatingDataSource,
    dsm.LinearPeriodicDataSource,
    dsm.LinearStepDataSource,
    dsm.LineDataSource,
    dsm.MixedBrownDriftDataSource,#15
    dsm.MixedDriftDataSource,
    dsm.PowDataSource,
    dsm.RandomUniformDataSource,
    dsm.RBFDriftDataSource,
    dsm.SinDriftDataSource,#20
    dsm.SineDataSource,
    dsm.SquareDataSource,
    dsm.StarDataSource,
    dsm.TimeBehaviorDataSource,
    dsm.ZDataSource,#25
    dsm.ZInvDataSource#26
]


shape_values = [(1,)]#, (1,1), (2,2), (5,3,2), (2,3,4,1)]
parameter_1 = [-2.7,-1,0,1,2.7]

@pytest.mark.parametrize("a", parameter_1)
@pytest.mark.parametrize("b", parameter_1)
@pytest.mark.parametrize("result_shape", shape_values)
@pytest.mark.parametrize("query_shape", shape_values)
def test_LineDataSource(query_shape, result_shape, a, b):
    bp = bps.BaselineBlueprint(process=DataSourceProcess(data_source=LineDataSource(query_shape=query_shape, result_shape=result_shape, a=a, b=b)),
                               evaluators=(tm.ResultEvaluator("process.data_source.query"),))
    er = ExperimentRunner([bp])
    er.run_experiment(bp)