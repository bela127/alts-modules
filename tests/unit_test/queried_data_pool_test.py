from alts.core.data.queried_data_pool import QueriedDataPool
import alts.modules.queried_data_pool as qdps

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
def auto_pass_test(qdp: QueriedDataPool):
    """
    | **Description**
    |   Automatically passes modules that are not tested for
    """
    if qdp in [qdps.FlatQueriedDataPool]:
        assert True
    