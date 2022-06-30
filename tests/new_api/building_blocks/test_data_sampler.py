import numpy as np

from alts.modules.queried_data_pool import FlatQueriedDataPool
from alts.modules.data_sampler import KDTreeKNNDataSampler
from alts.core.data.data_pool import DataPool
from alts.core.query.query_pool import QueryPool

from alts.core.experiment_modules import ExperimentModules

qp = QueryPool(None, query_shape=(2,), query_ranges=np.asarray([[0,1]]))
dp = DataPool(qp, result_shape=(2,))

def test_add_sample():

    qdp = FlatQueriedDataPool()
    qdp = qdp(qp, dp)

    em = ExperimentModules()
    em = em(queried_data_pool=qdp, oracle_data_pool=dp)

    sampler = KDTreeKNNDataSampler(10)
    sampler = sampler(em)

    query = np.asarray([(1,3)])
    result = np.asarray([(2,2)])

    qdp.add((query,result))

    queries, results = sampler.query(query)
    assert np.all(np.asarray((queries[0], results[0])) == np.asarray((query,result)))