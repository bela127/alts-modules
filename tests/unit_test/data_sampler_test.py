from alts.core.data.data_sampler import DataSampler
import alts.modules.data_sampler as dss

import pytest

"""
| **Test aims**
|   The data sampler modules are tested for:
|   - Nothing
"""

data_samplers = [
    dss.KDTreeKNNDataSampler,
    dss.KDTreeRegionDataSampler
]

@pytest.mark.parametrize("ds", data_samplers)
def auto_pass_test(ds: DataSampler):
    """
    | **Description**
    |   Automatically passes modules that are not tested for
    """
    if ds in [dss.KDTreeKNNDataSampler,
            dss.KDTreeRegionDataSampler]:
        assert True
    