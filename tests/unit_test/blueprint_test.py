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
def auto_pass_test(b: Blueprint):
    """
    | **Description**
    |   Automatically passes modules that are not tested for
    """
    if b in [bps.Blueprint]:
        assert True
    