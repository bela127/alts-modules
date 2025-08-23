from alts.core.data.queried_data_pool import QueriedDataPool
import alts.modules.queried_data_pool as qdps
from alts.core.data.constrains import QueryConstrain, ResultConstrain

import numpy as np
import pytest

"""
| **Test aims**
|   The queried data pool modules are tested for:
|   - Nothing
"""

queried_data_pools = [
    qdps.FlatQueriedDataPool
]

@pytest.mark.parametrize("qdp", queried_data_pools)
def test_add_query(qdp: QueriedDataPool):
    """
    | **Description**
    |   Tests for correct adding and querying of data points to the DataPool.
    """
    if qdp == qdps.FlatQueriedDataPool:
        query_shape = (1,)
        query_ranges = np.asarray(tuple((np.NINF, np.Inf) for i in range(query_shape[0])))
        qdp = qdp()(query_constrain=lambda:QueryConstrain(count=None, shape=query_shape, ranges=query_ranges), result_constrain=lambda:ResultConstrain(query_shape, query_ranges))
        assert qdp.query_index == {}, "QDP should be empty"
        #-----ADD-----
        #Single data point 1
        qdp.add((np.array([[1]]), np.array([[1]])))
        assert np.array_equal(qdp.query_index[tuple(np.array([1]))][0], np.array([1])), f"QDP should contain data point ([1],[1])"
        #Single data point 2
        qdp.add((np.array([[2]]), np.array([[2]])))
        assert np.array_equal(qdp.query_index[tuple(np.array([2]))][0], np.array([2])), f"QDP should contain data point ([2],[2])"
        #Multiple data points 1
        qdp.add((np.array([[3],[4],[5],[6]]), np.array([[6],[5],[4],[3]])))
        assert np.array_equal(qdp.query_index[tuple(np.array([3]))][0], np.array([6])), f"QDP should contain data points ([3],[6])"
        assert np.array_equal(qdp.query_index[tuple(np.array([4]))][0], np.array([5])), f"QDP should contain data points ([4],[5])"
        assert np.array_equal(qdp.query_index[tuple(np.array([5]))][0], np.array([4])), f"QDP should contain data points ([5],[4])"
        assert np.array_equal(qdp.query_index[tuple(np.array([6]))][0], np.array([3])), f"QDP should contain data points ([6],[3])"
        #Multiple data points 2
        qdp.add((np.array([[7],[8],[9]]), np.array([[-1],[0.5],[3]])))
        assert np.array_equal(qdp.query_index[tuple(np.array([7]))][0], np.array([-1])), f"QDP should contain data points ([7],[-1])"
        assert np.array_equal(qdp.query_index[tuple(np.array([8]))][0], np.array([0.5])), f"QDP should contain data points ([8],[0.5])"
        assert np.array_equal(qdp.query_index[tuple(np.array([9]))][0], np.array([3])), f"QDP should contain data points ([9],[3])"
        #Overlapping data points
        qdp.add((np.array([[3],[4],[5],[6]]), np.array([[3],[4],[5],[6]])))
        assert np.array_equal(qdp.query_index[tuple(np.array([3]))], [np.array([6]), np.array([3])]), f"QDP should contain data points ([3],[6]),([3],[3])"
        assert np.array_equal(qdp.query_index[tuple(np.array([4]))], [np.array([5]), np.array([4])]), f"QDP should contain data points ([4],[5]),([4],[4])"
        assert np.array_equal(qdp.query_index[tuple(np.array([5]))], [np.array([4]), np.array([5])]), f"QDP should contain data points ([5],[4]),([5],[5])"
        assert np.array_equal(qdp.query_index[tuple(np.array([6]))], [np.array([3]), np.array([6])]), f"QDP should contain data points ([6],[3]),([6],[6])"
        #-----QUERY-----
        #Single data point 1
        assert np.array_equal(qdp.query(np.array([[1]]))[1], np.array([[1]])), f"QDP should contain data point ([1],[1])"
        #Single data point 2
        assert np.array_equal(qdp.query(np.array([[7]]))[1], np.array([[-1]])), f"QDP should contain data point ([7],[-1])"
        #Multiple data points 1
        assert np.array_equal(qdp.query(np.array([[7],[8],[9]]))[1], np.array([[-1],[0.5],[3]])), f"QDP should contain data points ([7],[-1]),([8],[0.5]),([9],[3])"
        #Multiple data points 2
        pytest.xfail("How should QDP handle queries to non-existent data points?")
        assert np.array_equal(qdp.query(np.array([[0],[1],[1]]))[1], np.array([[1],[1]])), f"QDP should contain data point ([1],[1]), but nothing for [0]"
        #Overlapping data points 1
        res = qdp.query(np.array([[6]]))[1]
        assert np.array_equal(res, np.array([[3]])) or np.array_equal(res, np.array([[6]])), f"QDP should return data point ([3],[6]) or ([6],[3])"
        #Overlapping data points 2
        res = qdp.query(np.array([[6],[4]]))[1]
        assert np.array_equal(res[0], np.array([3])) or np.array_equal(res[0], np.array([6])), f"QDP should return data point ([3],[6]) or ([6],[3])"
        assert np.array_equal(res[1], np.array([4])) or np.array_equal(res[1], np.array([5])), f"QDP should return data point ([4],[5]) or ([4],[4])"
    else:
        raise ValueError("QueriedDataPool not found")
    #-----ADD-----
    #No data points
    #Single data point 1
    #Single data point 2
    #Multiple data points 1
    #Multiple data points 2
    #Overlapping data points
    #-----QUERY-----
    #No data points
    #Single data point 1
    #Single data point 2
    #Multiple data points 1
    #Multiple data points 2
    #Overlapping data points 1
    #Overlapping data points 2

@pytest.mark.parametrize("qdp", queried_data_pools)
def test_query(qdp: QueriedDataPool):
    pytest.skip("Not yet implemented")

@pytest.mark.parametrize("qdp", queried_data_pools)
def test_constraint(qdp: QueriedDataPool):
    pytest.skip("Not yet implemented")
    