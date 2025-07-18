#Test version 1.2 as of 18.07.2025
from alts.core.oracle.query_queue import QueryQueue
import alts.modules.oracle.query_queue as qq

from alts.core.configuration import Configurable, Required, is_set, pre_init, post_init, init
import alts.core.data.constrains as c

import numpy as np
import pytest

"""
| **Test aims**
|   The query queue modules are tested for:
|   - Correct setting of query constraints
|   - Adding queries
|   - Popping queries
|   - Properties: last, first, latest_add, latest_pop, empty, count
"""

query_queues = [
    qq.FCFSQueryQueue
]

@pytest.mark.parametrize("arg", query_queues)
def test_query_constrain(arg: type[QueryQueue]):
    """
    | **Description**
    |   Tests for the correct setting of its query constraint.
    """
    if arg == qq.FCFSQueryQueue:
        q: qq.FCFSQueryQueue = arg()(query_constrain=lambda : c.QueryConstrain(count=3, shape=(1,), ranges=np.array((-10,10))))
        q.post_init()
        assert q.query_constrain().count == 3, "Value should be as given above"
        assert q.query_constrain().shape == (1,), "Value should be as given above"
        assert (q.query_constrain().ranges == np.array((-10,10))).all(), "Value should be as given above"

@pytest.mark.parametrize("arg", query_queues)
def test_add(arg: type[QueryQueue]):
    """
    | **Description**
    |   Tests adding queries to the queue.
    """
    if arg == qq.FCFSQueryQueue:
        q: qq.FCFSQueryQueue = arg()(query_constrain=lambda : c.QueryConstrain(count=3, shape=(1,), ranges=np.array((-10,10))))
        q.post_init()
        q.add(np.array([[1]]))
        q.add(np.array([[1], [2], [3]]))
        assert (q.queries == np.array([[1], [1], [2], [3]])).all(), "Checkd if all added queries are present"

@pytest.mark.parametrize("arg", query_queues)
def test_pop(arg: type[QueryQueue]):
    """
    | **Description**
    |   Tests popping queries from the queue.
    """
    if arg == qq.FCFSQueryQueue:
        q: qq.FCFSQueryQueue = arg()(query_constrain=lambda : c.QueryConstrain(count=3, shape=(1,), ranges=np.array((-10,10))))
        q.post_init()
        q.add(np.array([[1]]))
        q.add(np.array([[1], [2], [3]]))
        assert (q.pop(3) == np.array([[1], [1], [2]])).all(), "Check if the correct queries have been popped"
        assert (q.pop() == np.array([[3]])), "Check if the correct query has been popped"

@pytest.mark.parametrize("arg", query_queues)
def test_properties(arg: type[QueryQueue]):
    """
    | **Description**
    |   Tests for correct functioning of all properties.
    """
    if arg == qq.FCFSQueryQueue:
        q: qq.FCFSQueryQueue = arg()(query_constrain=lambda : c.QueryConstrain(count=3, shape=(1,), ranges=np.array((-10,10))))
        q.post_init()
        assert (q.latest_add == np.array([])).all(), "Check latest_add, when nothing has been added"
        q.add(np.array([[1]]))
        q.add(np.array([[1], [2], [3]]))
        assert (q.last == np.array([3])).all(), "Check last attribute"
        assert (q.first == np.array([1])).all(), "Check first attribute"
        assert (q.latest_add == np.array([[1], [2], [3]])).all(), "Check latest_add attribute"
        assert (q.latest_pop == np.array([])).all(), "Check latest_pop, when nothing has been popped"
        assert not q.empty, "Check empty attribute"
        assert q.count == 4, "Check count attribute"
        q.pop(2)
        q.add(np.array([[7], [-1], [0]]))
        assert (q.last == np.array([0])).all(), "Check last attribute"
        assert (q.first == np.array([2])).all(), "Check first attribute"
        assert (q.latest_add == np.array([[7], [-1], [0]])).all(), "Check latest_add attribute"
        assert (q.latest_pop == np.array([[1], [1]])).all(), "Check latest_pop attribute"
        assert not q.empty, "Check empty attribute"
        assert q.count == 5, "Check count attribute"
        q.pop(5)
        assert (q.last == None).all() , "Check last attribute, when queue is empty"
        assert (q.first == None).all(), "Check first attribute, when queue is empty"
        assert (q.latest_add == np.array([[7], [-1], [0]])).all(), "Check latest_add attribute"
        assert (q.latest_pop == np.array([[2], [3], [7], [-1], [0]])).all(), "Check latest_pop attribute"
        assert q.empty, "Check empty attribute"
        assert q.count == 0, "Check count attribute"