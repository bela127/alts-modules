from alts.core.query.selection_criteria import SelectionCriteria
import alts.modules.query.selection_criteria as scm

import numpy as np
import pytest

#List of selection criterias
selection_criterias = [
    scm.AllSelectionCriteria,
    scm.NoSelectionCriteria,
    scm.RandomSelectionCriteria
]

#Test for single value
@pytest.mark.parametrize("sc", selection_criterias)
def test_single_value(sc: SelectionCriteria):
    sc = sc()
    if isinstance(sc, scm.AllSelectionCriteria):
        x,y = sc.query(np.array([[0,1]]))
        assert np.array_equal(x, np.array([[0,1]])) and y == np.ones((1,1))
    elif isinstance(sc, scm.NoSelectionCriteria):
        x,y = sc.query(np.array([[0,1]]))
        assert np.array_equal(x, np.array([[0,1]])) and y == np.zeros((1,1))
    elif isinstance(sc, scm.RandomSelectionCriteria):
        x,y = sc.query(np.array([[0,1]]))
        assert np.array_equal(x, np.array([[0,1]])) and y.shape == (1,1) and isinstance(y.flat[0], float) 

#Test for multiple values
@pytest.mark.parametrize("sc", selection_criterias)
def test_multiple_values(sc: SelectionCriteria):
    sc = sc()
    if isinstance(sc, scm.AllSelectionCriteria):
        x,y = sc.query(np.array([[0,-1],[2,0],[1,1],[3,2.6],[27,3]]))
        assert np.array_equal(x, np.array([[0,-1],[2,0],[1,1],[3,2.6],[27,3]])) and np.array_equal(y, np.ones((5,1)))
    elif isinstance(sc, scm.NoSelectionCriteria):
        x,y = sc.query(np.array([[0,-1],[2,0],[1,1],[3,2.6],[27,3]]))
        assert np.array_equal(x, np.array([[0,-1],[2,0],[1,1],[3,2.6],[27,3]])) and y == np.zeros((5,1))
    elif isinstance(sc, scm.RandomSelectionCriteria):
        x,y = sc.query(np.array([[0,-1],[2,0],[1,1],[3,2.6],[27,3]]))
        assert np.array_equal(x, np.array([[0,-1],[2,0],[1,1],[3,2.6],[27,3]])) and y.shape == (5,1) and isinstance(y.flat[0], float) 

#Test for multiple dimensions
@pytest.mark.parametrize("sc", selection_criterias)
def test_multiple_dimensions(sc: SelectionCriteria):
    sc = sc()
    if isinstance(sc, scm.AllSelectionCriteria):
        x,y = sc.query(np.array([[0,[-1,-1]],[2,[0,0]],[1,[1.1,1]],[3,[2.6,3.3]],[27,[3,100]]]))
        assert np.array_equal(x, np.array([[0,[-1,-1]],[2,[0,0]],[1,[1.1,1]],[3,[2.6,3.3]],[27,[3,100]]])) and y == np.ones((5,1))
    elif isinstance(sc, scm.NoSelectionCriteria):
        x,y = sc.query(np.array([[0,[-1,-1]],[2,[0,0]],[1,[1.1,1]],[3,[2.6,3.3]],[27,[3,100]]]))
        assert np.array_equal(x, np.array([[0,[-1,-1]],[2,[0,0]],[1,[1.1,1]],[3,[2.6,3.3]],[27,[3,100]]])) and y == np.zeros((5,1))
    elif isinstance(sc, scm.RandomSelectionCriteria):
        x,y = sc.query(np.array([[0,[-1,-1]],[2,[0,0]],[1,[1.1,1]],[3,[2.6,3.3]],[27,[3,100]]]))
        assert np.array_equal(x, np.array([[0,[-1,-1]],[2,[0,0]],[1,[1.1,1]],[3,[2.6,3.3]],[27,[3,100]]])) and y.shape == (5,1) and isinstance(y.flat[0], float) 