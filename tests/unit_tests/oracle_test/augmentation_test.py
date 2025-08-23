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
def test_constrains(a: type[Augmentation]):
    """
    | **Description**
    |   Checks if the Augmentation stays within its constraints.
    """
    if a == am.NoiseAugmentation:
        pytest.skip("Constrain")
    else:
        raise ValueError(f"Augmentation not found: {a}")
    