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
        assert q.query_constrain().count == 3
        assert q.query_constrain().shape == (1,)
        assert (q.query_constrain().ranges == np.array((-10,10))).all()

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
        assert (q.queries == np.array([[1], [1], [2], [3]])).all()

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
        assert (q.pop(3) == np.array([[1], [1], [2]])).all()
        assert (q.pop() == np.array([[3]]))

@pytest.mark.parametrize("arg", query_queues)
def test_properties(arg: type[QueryQueue]):
    """
    | **Description**
    |   Tests for correct functioning of all properties.
    """
    if arg == qq.FCFSQueryQueue:
        q: qq.FCFSQueryQueue = arg()(query_constrain=lambda : c.QueryConstrain(count=3, shape=(1,), ranges=np.array((-10,10))))
        q.post_init()
        q.add(np.array([[1]]))
        q.add(np.array([[1], [2], [3]]))
        assert (q.last == np.array([3])).all()
        assert (q.first == np.array([1])).all()
        assert (q.latest_add == np.array([3])).all()
        #assert q.latest_pop == None
        assert not q.empty
        assert q.count == 4
        q.pop(2)
        q.add(np.array([[7], [-1], [0]]))
        assert (q.last == np.array([0])).all()
        assert (q.first == np.array([2])).all()
        assert (q.latest_add == np.array([0])).all()
        assert (q.latest_pop == np.array([[1], [1]])).all()
        assert not q.empty
        assert q.count == 5
        q.pop(5)
        assert (q.last == None).all()
        assert (q.first == None).all()
        assert (q.latest_add == np.array([0])).all()
        assert (q.latest_pop == np.array([[2], [3], [7], [-1], [0]])).all()
        assert q.empty
        assert q.count == 0