from alts.core.query.selection_criteria import SelectionCriteria
import alts.modules.query.selection_criteria as scm

import numpy as np
import pytest

"""
| **Test aims**
|   The selection criteria modules are tested for:
|   - single queries
|   - multiple queries
|   - multidimensional queries
"""

#List of selection criterias
selection_criterias = [
    scm.AllSelectionCriteria,
    scm.NoSelectionCriteria,
    scm.RandomSelectionCriteria
]

#Test for single value
@pytest.mark.parametrize("sc", selection_criterias)
def test_single_value(sc: SelectionCriteria):
    """
    | **Description**
    |   To pass this test, the module has to correctly select from the single query [1].
    """
    sc = sc()
    if isinstance(sc, scm.AllSelectionCriteria):
        x,y = sc.query(np.array([1]))
        assert np.array_equal(x, np.array([1])) and np.array_equal(y, np.ones((1,1)))
    elif isinstance(sc, scm.NoSelectionCriteria):
        x,y = sc.query(np.array([1]))
        assert np.array_equal(x, np.array([1])) and np.array_equal(y, np.zeros((1,1)))
    elif isinstance(sc, scm.RandomSelectionCriteria):
        x,y = sc.query(np.array([1]))
        assert np.array_equal(x, np.array([1])) and y.shape == (1,1) and isinstance(y.flat[0], float) 

#Test for multiple values
@pytest.mark.parametrize("sc", selection_criterias)
def test_multiple_queries(sc: SelectionCriteria):
    """
    | **Description**
    |   To pass this test, the module has to correctly select from a list of 1-D queries.
    """
    sc = sc()
    if isinstance(sc, scm.AllSelectionCriteria):
        x,y = sc.query(np.array([[0,-1],[2,0],[1,1],[3,2.6],[27,3]]))
        assert np.array_equal(x, np.array([[0,-1],[2,0],[1,1],[3,2.6],[27,3]])) and np.array_equal(y, np.ones((5,1)))
    elif isinstance(sc, scm.NoSelectionCriteria):
        x,y = sc.query(np.array([[0,-1],[2,0],[1,1],[3,2.6],[27,3]]))
        assert np.array_equal(x, np.array([[0,-1],[2,0],[1,1],[3,2.6],[27,3]])) and np.array_equal(y, np.zeros((5,1)))
    elif isinstance(sc, scm.RandomSelectionCriteria):
        x,y = sc.query(np.array([[0,-1],[2,0],[1,1],[3,2.6],[27,3]]))
        assert np.array_equal(x, np.array([[0,-1],[2,0],[1,1],[3,2.6],[27,3]])) and y.shape == (5,1) and isinstance(y.flat[0], float)

#Test for multiple dimensions
@pytest.mark.parametrize("sc", selection_criterias)
def test_multiple_dimensions(sc: SelectionCriteria):
    """
    | **Description**
    |   To pass this test, the module has to correctly select from a list of 1-D queries.
    """
    sc = sc()
    if isinstance(sc, scm.AllSelectionCriteria):
        x,y = sc.query(np.array([[[2,3],[-1,-1]],[[0,0],[0,0]],[[5.1,3],[1.1,1]],[[-1,-3],[2.6,3.3]],[[1,-5],[3,100]]]))
        assert np.array_equal(x, np.array([[[2,3],[-1,-1]],[[0,0],[0,0]],[[5.1,3],[1.1,1]],[[-1,-3],[2.6,3.3]],[[1,-5],[3,100]]])) and np.array_equal(y, np.ones((5,1)))
    elif isinstance(sc, scm.NoSelectionCriteria):
        x,y = sc.query(np.array([[[2,3],[-1,-1]],[[0,0],[0,0]],[[5.1,3],[1.1,1]],[[-1,-3],[2.6,3.3]],[[1,-5],[3,100]]]))
        assert np.array_equal(x, np.array([[[2,3],[-1,-1]],[[0,0],[0,0]],[[5.1,3],[1.1,1]],[[-1,-3],[2.6,3.3]],[[1,-5],[3,100]]])) and np.array_equal(y, np.zeros((5,1)))
    elif isinstance(sc, scm.RandomSelectionCriteria):
        x,y = sc.query(np.array([[[2,3],[-1,-1]],[[0,0],[0,0]],[[5.1,3],[1.1,1]],[[-1,-3],[2.6,3.3]],[[1,-5],[3,100]]]))
        assert np.array_equal(x, np.array([[[2,3],[-1,-1]],[[0,0],[0,0]],[[5.1,3],[1.1,1]],[[-1,-3],[2.6,3.3]],[[1,-5],[3,100]]])) and y.shape == (5,1) and isinstance(y.flat[0], float) 