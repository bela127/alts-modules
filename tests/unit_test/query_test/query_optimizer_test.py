from alts.core.query.query_optimizer import QueryOptimizer
import alts.modules.query.query_optimizer as qos

import pytest

"""
| **Test aims**
|   The query optimizers modules are tested for:
|   - Nothing
"""

query_optimizers = [
    qos.NoQueryOptimizer,
    qos.GAQueryOptimizer,
    qos.MCQueryOptimizer,
    qos.MaxMCQueryOptimizer,
    qos.ProbWeightedMCQueryOptimizer
]

@pytest.mark.parametrize("qo", query_optimizers)
def test_auto_skip(qo: QueryOptimizer):
    """
    | **Description**
    |   Automatically skips modules that are not tested for
    """
    if qo in [qos.NoQueryOptimizer,
            qos.GAQueryOptimizer,
            qos.MCQueryOptimizer,
            qos.MaxMCQueryOptimizer,
            qos.ProbWeightedMCQueryOptimizer]:
        pytest.skip("Not yet implemented")
    