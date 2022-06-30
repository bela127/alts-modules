import numpy as np

from alts.modules.queried_data_pool import FlatQueriedDataPool
from alts.core.data_subscriber import DataSubscriber

from alts.core.data.data_pool import DataPool
from alts.core.query.query_pool import QueryPool

qp = QueryPool(None, query_shape=(1,), query_ranges=np.asarray([[0,1]]))
dp = DataPool(qp, result_shape=(2,))

def test_data_pool_add():
    qdp = FlatQueriedDataPool()
    qdp = qdp(qp, dp)
    assert qdp.data_pool.result_shape == (2,)

    queries = np.asarray(((1,),))
    results = np.asarray(((2,2),))

    qdp.add((queries, results))
    assert np.all(qdp.queries[0] == queries[0])
    assert np.all(qdp.results[0] == results[0])


def test_data_pool_subscribe():
    qdp = FlatQueriedDataPool()
    qdp = qdp(qp, dp)

    queries = np.asarray(((1,),))
    results = np.asarray(((2,2),))

    class Test_Subscriber(DataSubscriber):
        def update(self, data_point):
            queries_u, results_u = data_point
            assert np.all(queries_u == queries)
            assert np.all(results_u == results)
            
    sub = Test_Subscriber()
    sub = sub()
    qdp.subscribe(sub)

    assert qdp._subscriber[0] == sub

    qdp.add((queries,results))