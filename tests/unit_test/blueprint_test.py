from alts.core.blueprint import Blueprint
import alts.modules.blueprint as bps

import pytest

"""
| **Test aims**
|   The blueprint modules are tested for:
|   - Nothing
"""

blueprints = [
    bps.Blueprint
]

@pytest.mark.parametrize("b", blueprints)
def test_auto_skip(b: Blueprint):
    """
    | **Description**
    |   Automatically skips modules that are not tested for
    """
    if b in [bps.Blueprint]:
        pytest.skip("Not yet implemented")
    