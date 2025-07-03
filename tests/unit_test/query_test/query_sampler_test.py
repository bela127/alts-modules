from alts.core.query.query_sampler import QuerySampler
import alts.modules.query.query_sampler as qss

import numpy as np
import pytest

"""
| **Test aims**
|   The query samplers modules are tested for:
|   - Correct sampling of data 1D
|   - Correct sampling of data 2D
"""

query_samplers = [
    qss.OptimalQuerySampler,
    qss.FixedQuerySampler,
    qss.FixedQuerySampler,
    qss.UniformQuerySampler,
    qss.LatinHypercubeQuerySampler,
    qss.RandomChoiceQuerySampler,
    qss.ProcessQuerySampler,
    qss.LastProcessQuerySampler,
    qss.ProcessQueueQuerySampler,
    qss.DataPoolQuerySampler,
    qss.AllDataPoolQuerySampler,
    qss.AllResultPoolQuerySampler,
    qss.AllStreamPoolQuerySampler,
    qss.AllProcessPoolQuerySampler,
    qss.LastDataPoolQuerySampler,
    qss.LastResultPoolQuerySampler,
    qss.LastStreamPoolQuerySampler,
    qss.LastProcessPoolQuerySampler
]

queries = np.array([[0],[0.25],[0.5],[0.75],[1]])

@pytest.mark.parametrize("qs", query_samplers)
def test_1d_sample(qs: QuerySampler):
    if qs == qss.OptimalQuerySampler:
        qs = qs(optimal_queries=queries)
        
    
@pytest.mark.parametrize("qs", query_samplers)
def test_2d_sample(qs: QuerySampler):
    pytest.skip("Not implemented")