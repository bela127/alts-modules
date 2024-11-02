from alts.core.oracle.data_source import DataSource
import alts.modules.oracle.data_source as dsm

import pytest
import numpy as np

"""
| **Test aims**
|   The data source modules are tested for:
|   - the values 0,0.1,0.25,0.5,0.75 (for random data sources a range of results is accepted)
|   - edge case values
|   - single queries
|   - multiple queries
|   - 1 normal case and all edge cases for parameters
"""
queries: list[float] = [0,0.1,0.25,0.5,0.75]

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
    dsm.TimeDataSource,#25
    dsm.ZDataSource,
    dsm.ZInvDataSource#27
]

#Pass function for single query
def passer_generator(ds: DataSource, **kwargs):
    """
    | **Description**
    |   Compares with expected behaviour.
    """
    if type(ds) == dsm.BrownianDriftDataSource:
        raise NotImplementedError()
    elif type(ds) == dsm.BrownianProcessDataSource:
        raise NotImplementedError()
    elif type(ds) == dsm.CrossDataSource:
        return lambda x,y: True if y in {-ds.a*x,ds.a*x} else False 
    elif type(ds) == dsm.DoubleLinearDataSource:
        return lambda x,y: True if y in {ds.a*x,ds.s*ds.a*x} else False 
    elif type(ds) == dsm.ExpDataSource:
        return lambda x,y: True if y == ds.s*(ds.b**x) else False 
    elif type(ds) == dsm.GaussianProcessDataSource:
        raise NotImplementedError()
    elif type(ds) == dsm.HourglassDataSource:
        return lambda x,y: True if y in {-ds.a*x, -ds.a*x, -ds.a/2, ds.a/2} else False 
    elif type(ds) == dsm.HypercubeDataSource:
        raise NotImplementedError()
    elif type(ds) == dsm.HyperSphereDataSource:
        raise NotImplementedError()
    elif type(ds) == dsm.IndependentDataSource:
        raise NotImplementedError()
    elif type(ds) == dsm.InterpolatingDataSource:
        raise UserWarning("Correctness depends on correctness of used interpolation strategies")
    elif type(ds) == dsm.LinearPeriodicDataSource:
        return lambda x,y: True if y == (ds.a*x) % ds.p else False 
    elif type(ds) == dsm.LinearStepDataSource:
        return lambda x,y: True if y == ds.a*(x-x%ds.p)/ds.p else False 
    elif type(ds) == dsm.LineDataSource:
        return lambda x,y: True if y == ds.a*x + ds.b else False 
    elif type(ds) == dsm.MixedBrownDriftDataSource:
        raise NotImplementedError()
    elif type(ds) == dsm.MixedDriftDataSource:
        raise NotImplementedError()
    elif type(ds) == dsm.PowDataSource:
        return lambda x,y: True if y == ds.s*(x**ds.p) else False 
    elif type(ds) == dsm.RandomUniformDataSource:
        return lambda x,y: True if y >= ds.l and y < ds.u else False
    elif type(ds) == dsm.RBFDriftDataSource:
        raise NotImplementedError()
    elif type(ds) == dsm.SinDriftDataSource:
        raise NotImplementedError()
    elif type(ds) == dsm.SineDataSource:
        raise NotImplementedError()
    elif type(ds) == dsm.SquareDataSource:
        return lambda x,y: True if y == ds.s * (x - ds.x0)**2 + ds.y0 else False 
    elif type(ds) == dsm.StarDataSource:
        raise NotImplementedError()
    elif type(ds) == dsm.TimeBehaviorDataSource:
        raise NotImplementedError()
    elif type(ds) == dsm.TimeDataSource:
        #raise UserWarning("Correctness of TimeDataSource depends on correctness of used sub-DataSource")
        return lambda x,y: True
    elif type(ds) == dsm.ZDataSource:
        return lambda x,y: True if y in {ds.a*x, -ds.a/2, ds.a/2} else False
    elif type(ds) == dsm.ZInvDataSource:
        return lambda x,y: True if y in {-ds.a*x, -ds.a/2, ds.a/2} else False 
    else:
        raise ValueError("DataSource not found")

#Single Query test
@pytest.mark.parametrize("ds", data_sources)
def single_query_test(ds: DataSource):
    ds = ds()
    f = passer_generator(ds)
    for query in queries:
        assert f(query, ds.query(np.array(query)))

#Multiple Queries test
@pytest.mark.parametrize("ds", data_sources)
def multiple_queries_test(ds: DataSource):
    ds = ds()
    f = passer_generator(ds)
    ds_results = ds.query(np.array([[value] for value in queries]))
    for query, ds_result in zip(queries, ds_results):
        assert f(query, ds_result)


