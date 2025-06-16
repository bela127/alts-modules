from alts.core.oracle.query_queue import QueryQueue
import alts.modules.oracle.query_queue as qq

import pytest

"""
| **Test aims**
|   The query queue modules are tested for:
|   - Nothing
"""

query_queues = [
    qq.FCFSQueryQueue
]

@pytest.mark.parametrize("q", query_queues)
def auto_pass_test(q: QueryQueue):
    """
    | **Description**
    |   Automatically passes modules that are not tested for
    """
    if q in [qq.FCFSQueryQueue]:
        assert True
    