from alts.core.query.query_sampler import QuerySampler
import alts.modules.query.query_sampler as qss

import pytest

"""
| **Test aims**
|   The query samplers modules are tested for:
|   - Nothing
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

@pytest.mark.parametrize("qs", query_samplers)
def test_auto_skip(qs: QuerySampler):
    """
    | **Description**
    |   Automatically skips modules that are not tested for
    """
    if qs in [qss.OptimalQuerySampler,
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
            qss.LastProcessPoolQuerySampler]:
        pytest.skip("Not yet implemented")
    