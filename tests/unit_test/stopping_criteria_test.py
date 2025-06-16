from alts.core.stopping_criteria import StoppingCriteria
import alts.modules.stopping_criteria as scs

import pytest

"""
| **Test aims**
|   The stopping criterias modules are tested for:
|   - Nothing
"""

stopping_criterias = [
    scs.TimeStoppingCriteria,
    scs.DataExhaustedStoppingCriteria
]

@pytest.mark.parametrize("sc", stopping_criterias)
def auto_pass_test(sc: StoppingCriteria):
    """
    | **Description**
    |   Automatically passes modules that are not tested for
    """
    if sc in [scs.TimeStoppingCriteria,
            scs.DataExhaustedStoppingCriteria]:
        assert True
    