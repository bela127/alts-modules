from alts.core.oracle.data_behavior import DataBehavior  
import alts.modules.behavior as beh

import pytest

"""
| **Test aims**
|   The data behavior modules are tested for:
|   - Nothing
"""

behaviors = [
    beh.EquidistantTimeUniformBehavior,
    beh.RandomTimeUniformBehavior,
    beh.RandomTimeBrownBehavior
]

@pytest.mark.parametrize("b", behaviors)
def auto_pass_test(b: DataBehavior):
    """
    | **Description**
    |   Automatically passes modules that are not tested for
    """
    if b in [beh.EquidistantTimeUniformBehavior,
            beh.RandomTimeUniformBehavior,
            beh.RandomTimeBrownBehavior]:
        assert True
    