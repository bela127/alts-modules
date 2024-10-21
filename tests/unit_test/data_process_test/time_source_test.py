from alts.core.data_process.time_source import TimeSource
import alts.modules.data_process.time_source as tsm

import pytest

"""
| **Test aims**
|   The time source modules are tested for:
|   - time setting
|   - time stepping
"""

time_sources = [
    tsm.IterationTimeSource
]

#Time setting
@pytest.mark.parametrize("ts", time_sources)
def test_time_setting(ts: TimeSource):
    """
    | **Description**
    |   To pass, the TimeSource has has to correctly set and then display the time when modified.
    """
    ts = ts()()
    #IterationTimeSource
    if isinstance(ts, tsm.IterationTimeSource):  
        #Two simple tests for time setting/adding
        ts.time = 5
        assert ts.time == 5
        ts.time = 10
        assert ts.time == 15
        #Error expected as value is non-positive
        try:
            ts.time = 0
        except Exception as e:
            assert isinstance(e, ValueError)
        else:
            assert False
        try:
            ts.time = -3
        except Exception as e:
            assert isinstance(e, ValueError)
        else:
            assert False
        #Final addition test
        ts.time = 2
        assert ts.time == 17
    else:
        raise ValueError("TimeSource not found: {}".format(ts))

#Stepping
@pytest.mark.parametrize("ts", time_sources)
def test_step_read(ts: TimeSource):
    """
    | **Description**
    |   To pass, the TimeSource has has to correctly step and display the time 5 times.
    |   Depends on the passing of the time_setting test.
    """
    if ts == tsm.IterationTimeSource:
        #Normal step
        ts = ts(3,2)() # type: ignore
        ts.post_init()
        ts.step(1)
        assert ts.time == 5
        ts.step(5)
        assert ts.time == 13
        ts.step(3)
        assert ts.time == 9
        #Float steps
        ts = tsm.IterationTimeSource(1.5,1.2)() # type: ignore
        ts.post_init()
        ts.step(3)
        assert ts.time == 5.1
        #Step after setting time
        ts = tsm.IterationTimeSource(3,2)() # type: ignore
        ts.post_init()
        ts.step(1)
        assert ts.time == 5
        ts.time = 4
        ts.step(2)
        assert ts.time == 11
        #Edge case: step-length 0
        ts = tsm.IterationTimeSource(0,0)() # type: ignore
        ts.post_init()
        ts.step(3)
        assert ts.time == 0
        #Edge case: 0 steps
        ts = tsm.IterationTimeSource(5,3)() # type: ignore
        ts.post_init()
        ts.step(0)
        assert ts.time == 5
    else:
        raise ValueError("TimeSource not found: {}".format(ts))
