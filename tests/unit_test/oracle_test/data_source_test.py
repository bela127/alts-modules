
from alts.core.oracle.data_source import DataSource
import alts.modules.oracle.data_source as dsm

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
    dsm.ZDataSource,#25
    dsm.ZInvDataSource#26
]

#Pass function for single query
def passer_generator(ds: DataSource, **kwargs):
    """
    | **Description**
    |   Compares with expected behaviour.
    """
    if type(ds) == dsm.BrownianDriftDataSource:
        pytest.skip("Not implemented")
    elif type(ds) == dsm.BrownianProcessDataSource:
        pytest.skip("Not implemented")
    elif type(ds) == dsm.CrossDataSource:
        return lambda x,y: True if y in {-ds.a*x,ds.a*x} else False 
    elif type(ds) == dsm.DoubleLinearDataSource:
        return lambda x,y: True if y in {ds.a*x,ds.s*ds.a*x} else False 
    elif type(ds) == dsm.ExpDataSource:
        return lambda x,y: True if y == ds.s*(ds.b**x) else False 
    elif type(ds) == dsm.GaussianProcessDataSource:
        pytest.skip("Not implemented")
    elif type(ds) == dsm.HourglassDataSource:
        return lambda x,y: True if y in {-ds.a*x, ds.a*x, -ds.a/2, ds.a/2} else False 
    elif type(ds) == dsm.HypercubeDataSource:
        return lambda x,y: True if -ds.w <= x < ds.w and y in {-0.5, 0.5} or (x < -ds.w or x >= ds.w) and -0.5 <= y < 0.5 else False
    elif type(ds) == dsm.HyperSphereDataSource:
        return lambda x,y: True if y in {-np.sqrt(np.abs(1-x*x)), np.sqrt(np.abs(1-x*x))} else False
    elif type(ds) == dsm.IndependentDataSource:
        pytest.skip("Not implemented")
    elif type(ds) == dsm.InterpolatingDataSource:
        pytest.skip("Not implemented")
    elif type(ds) == dsm.LinearPeriodicDataSource:
        return lambda x,y: True if y == (ds.a*x) % ds.p else False 
    elif type(ds) == dsm.LinearStepDataSource:
        return lambda x,y: True if y == ds.a*(x-x%ds.p)/ds.p else False 
    elif type(ds) == dsm.LineDataSource:
        return lambda x,y: True if y == ds.a*x + ds.b else False 
    elif type(ds) == dsm.MixedBrownDriftDataSource:
        pytest.skip("Not implemented")
    elif type(ds) == dsm.MixedDriftDataSource:
        pytest.skip("Not implemented")
    elif type(ds) == dsm.PowDataSource:
        return lambda x,y: True if y == ds.s*(x**ds.p) else False 
    elif type(ds) == dsm.RandomUniformDataSource:
        return lambda x,y: True if y >= ds.l and y < ds.u else False
    elif type(ds) == dsm.RBFDriftDataSource:
        pytest.skip("Not implemented")
    elif type(ds) == dsm.SinDriftDataSource:
        pytest.skip("Not implemented")
    elif type(ds) == dsm.SineDataSource:
        return lambda x,y: True if y == np.sin((x - ds.x0) * 2 * np.pi * ds.p) + ds.y0 else False
    elif type(ds) == dsm.SquareDataSource:
        return lambda x,y: True if y == ds.s * (x - ds.x0)**2 + ds.y0 else False 
    elif type(ds) == dsm.StarDataSource:
        return lambda x,y: True if -ds.w <= x < ds.w and -0.5 <= y < 0.5 or (x < -ds.w or x >= ds.w) and y in {-x, 0, x} else False
    elif type(ds) == dsm.TimeBehaviorDataSource:
        pytest.skip("Not implemented")
    elif type(ds) == dsm.ZDataSource:
        return lambda x,y: True if y in {ds.a*x, -ds.a/2, ds.a/2} else False
    elif type(ds) == dsm.ZInvDataSource:
        return lambda x,y: True if y in {-ds.a*x, -ds.a/2, ds.a/2} else False 
    else:
        raise ValueError("DataSource not found")

#Single 1D Query test
@pytest.mark.parametrize("ds", data_sources)
def test_single_query(ds: DataSource):
    ds = ds()
    f = passer_generator(ds)
    for query in queries:
        y = ds.query(np.array([[query]]))
        print(f"{ds.__class__}: {ds.query_shape}, {y[0][0].shape}, {ds.result_shape}, {y[1][0].shape}, {y[0]}, {y[1]}")
        assert f(y[0][0][0], y[1][0][0])

#Multiple 1D Queries test
@pytest.mark.parametrize("ds", data_sources)
def test_multiple_queries(ds: DataSource):
    ds = ds()
    f = passer_generator(ds)
    ds_results = ds.query(np.array([[value] for value in queries]))
    for query, ds_result in zip(ds_results[0], ds_results[1]):
        print(f"{ds.__class__}: {ds.query_shape}, {query.shape}, {ds.result_shape}, {ds_result.shape}, {query}, {ds_result}")
        assert f(query[0], ds_result[0])


#Multiple 2D Queries test
@pytest.mark.parametrize("ds", data_sources)
def test_multiple_2d_queries(ds: DataSource):
    pytest.skip("WIP")
    ds = ds(query_shape=(2,2), result_shape=(2,2))()
    f = passer_generator(ds)
    ds_results = ds.query(np.array([[[queries[0], queries[1]],[queries[2], queries[3]]],[[queries[-1], queries[-2]],[queries[-3], queries[-4]]]]))
    for query, ds_result in zip(ds_results[0].flat, ds_results[1].flat):
        print(f"{ds.__class__}: {query.shape}, {ds_result.shape}, {query}, {ds_result}")
        assert f(query, ds_result)

#Different datasource parameters test
@pytest.mark.parametrize("ds", data_sources)
def test_parameters(ds: DataSource):
    pytest.skip("Not implemented")

