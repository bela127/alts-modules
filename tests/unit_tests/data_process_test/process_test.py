from alts.core.data_process.process import Process
import alts.modules.data_process.process as pr

import pytest

"""
| **Test aims**
|   The processes modules are tested for:
|   - Nothing
"""

processes = [
    pr.StreamProcess,
    pr.DataSourceProcess,
    pr.DelayedProcess,
    pr.DelayedStreamProcess,
    pr.IntegratingDSProcess,    
    pr.WindowDSProcess
]

@pytest.mark.parametrize("p", processes)
def test_auto_skip(p: Process):
    """
    | **Description**
    |   Automatically skips modules that are not tested for
    """
    if p in [pr.StreamProcess,
            pr.DataSourceProcess,
            pr.DelayedProcess,
            pr.DelayedStreamProcess,
            pr.IntegratingDSProcess,    
            pr.WindowDSProcess]:
        pytest.skip("Not yet implemented")
    