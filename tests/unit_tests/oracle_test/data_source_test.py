import alts.modules.oracle.data_source as dsm

from alts.modules.data_process.process import DataSourceProcess
import alts.modules.blueprint as bps
from alts.core.experiment_runner import ExperimentRunner
from alts.core.oracle.data_source import DataSource

import alts.modules.testing_modules as tm
import pytest
import numpy as np

"""
| **Test aims**
|   The query deciders are tested for:
|   - Correct output dimensions with respect to paramters
|   - Handling of nonsensical user parameters
"""

shape_values = [(1,1), (2,2)]#, (1,), (5,3,2), (2,3,4,1)]
parameter_any = [-2.7,-1]#,0,1,2.7]
parameter_positve = [1, 2.7]
parameter_nonpositive = [-2.7, -1]


@pytest.mark.parametrize("a", parameter_any)
@pytest.mark.parametrize("b", parameter_any)
@pytest.mark.parametrize("result_shape", shape_values)
@pytest.mark.parametrize("query_shape,", shape_values)
@pytest.mark.xfail(reason="Fix bad result shapes")
def test_LineDataSource(query_shape, result_shape, a, b):
    bp = bps.BaselineBlueprint(process=DataSourceProcess(data_source=dsm.LineDataSource(query_shape=query_shape, result_shape=result_shape, a=a, b=b)),
                               evaluators=(tm.ConstraintEvaluator(func_path="process.data_source.query", q_index=slice(None,None,None), r_index=1),))
    er = ExperimentRunner([bp])
    er.run_experiment(bp)

@pytest.mark.parametrize("u", parameter_nonpositive)
@pytest.mark.parametrize("l", parameter_any)
@pytest.mark.parametrize("result_shape", shape_values)
@pytest.mark.parametrize("query_shape,", shape_values)
@pytest.mark.xfail()
def test_RandomUniformDataSource(query_shape, result_shape, u, l):
    bp = bps.BaselineBlueprint(process=DataSourceProcess(data_source=dsm.RandomUniformDataSource(query_shape=query_shape, result_shape=result_shape, u=u+l, l=l)),
                               evaluators=(tm.ConstraintEvaluator(func_path="process.data_source.query", q_index=slice(None,None,None), r_index=1),))
    er = ExperimentRunner([bp])
    er.run_experiment(bp)


data_sources = tm.query_for_members(dsm, DataSource)

#List of DataSources
tested_data_sources = {
    dsm.BrownianDriftDataSource : None,
    dsm.BrownianProcessDataSource : None,
    dsm.CrossDataSource : None,
    dsm.DoubleLinearDataSource : None,
    dsm.ExpDataSource : None,#5
    dsm.GaussianProcessDataSource : None,
    dsm.HourglassDataSource : None,
    dsm.HypercubeDataSource : None,
    dsm.HyperSphereDataSource : None,
    dsm.IndependentDataSource : None,#10
    dsm.InterpolatingDataSource : None,
    dsm.LinearPeriodicDataSource : None,
    dsm.LinearStepDataSource : None,
    dsm.LineDataSource : test_LineDataSource,
    dsm.MixedBrownDriftDataSource : None,#15
    dsm.MixedDriftDataSource : None,
    dsm.PowDataSource : None,
    dsm.RandomUniformDataSource : None,
    dsm.RBFDriftDataSource : None,
    dsm.SinDriftDataSource : None,#20
    dsm.SineDataSource : None,
    dsm.SquareDataSource : None,
    dsm.StarDataSource : None,
    dsm.TimeBehaviorDataSource : None,
    dsm.ZDataSource : None,#25
    dsm.ZInvDataSource : None#26
}

@pytest.mark.parametrize("untested_data_source", np.setdiff1d(data_sources, list(tested_data_sources.keys()), assume_unique=True))
def test_special_tested(untested_data_source: type[DataSource]):
    pytest.xfail(f"DataSource {untested_data_source.__name__} not tested")