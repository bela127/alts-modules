from alts.core.data_process.time_source import TimeSource
import alts.modules.data_process.time_source as tsm

import numpy as np
import pytest

time_sources = [
    tsm.IterationTimeSource
]

#Time setting
@pytest.mark.parametrize("ts", time_sources)
def test_time_setting(ts: TimeSource):
    """
    | **Description**
    |   To pass, the TimeSource has has to correctly set the time when modified.
    """
    ts = ts()
    if type(ts) == tsm.IterationTimeSource:   
        ts.time = 5
        assert ts.time == 5
        ts.time = 10
        assert ts.time == 15
        try:
            ts.time = 0
        except Exception as e:
            assert type(e) == ValueError
        else:
            assert False
        try:
            ts.time = -3
        except Exception as e:
            assert type(e) == ValueError
        else:
            assert False
        ts.time = 2
        assert ts.time == 17
    else:
        raise ValueError("TimeSource not found: {}".format(ts))

#TODO Doing steps
#TODO Reading time

