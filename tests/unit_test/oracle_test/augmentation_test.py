from alts.core.oracle.augmentation import Augmentation
import alts.modules.oracle.augmentation as am

import pytest

"""
| **Test aims**
|   The augmentation modules are tested for:
|   - Nothing
|   Augmentations will not be tested for randomness
"""

augmentations = [
    am.NoiseAugmentation
]

@pytest.mark.parametrize("a", augmentations)
def auto_pass_test(a: Augmentation):
    """
    | **Description**
    |   Automatically passes modules that are not tested for
    """
    if a in [am.NoiseAugmentation]:
        assert True
    