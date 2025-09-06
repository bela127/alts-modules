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
|   - WIP
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


shape_values = [(1,1), (2,2)]#, (1,), (5,3,2), (2,3,4,1)]
parameter_1 = [-2.7,-1]#,0,1,2.7]

def query(func, queries):
    print("hi")
    results = np.dot(queries, np.ones((1,1))*(-2.7)) + np.ones((1,1))*(-2.7)
    print(queries.shape)
    print(np.ones((*(1,1),*(1,1))).shape)
    print(np.dot(queries, np.ones((*(1,1),*(1,1)))*(-2.7)).shape)
    print(np.ones((1,1)).shape)
    return queries, results

@pytest.mark.parametrize("a", parameter_1)
@pytest.mark.parametrize("b", parameter_1)
@pytest.mark.parametrize("result_shape", shape_values)
@pytest.mark.parametrize("query_shape,", shape_values)
def test_LineDataSource(query_shape, result_shape, a, b):
    bp = bps.BaselineBlueprint(process=DataSourceProcess(data_source=LineDataSource(query_shape=query_shape, result_shape=result_shape, a=a, b=b)),
                               evaluators=(tm.ConstrainEvaluator(func_path="process.data_source.query", q_index=slice(None,None,None), r_index=1),))
    er = ExperimentRunner([bp])
    er.run_experiment(bp)