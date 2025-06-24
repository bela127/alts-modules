from alts.core.evaluator import Evaluator
import alts.modules.evaluator as evs

import pytest

"""
| **Test aims**
|   The evaluator modules are tested for:
|   - Nothing
"""

evaluators = [
    evs.PrintNewDataPointsEvaluator,
    evs.PrintQueryEvaluator,
    evs.PrintExpTimeEvaluator,
    evs.PrintTimeSourceEvaluator,
    evs.PlotNewDataPointsEvaluator,
    evs.PlotAllDataPointsEvaluator,
    evs.PlotQueryDistEvaluator,
    evs.PlotSampledQueriesEvaluator,
    evs.LogOracleEvaluator,
    evs.LogStreamEvaluator,
    evs.LogProcessEvaluator,
    evs.LogResultEvaluator,
    evs.LogAllEvaluator,
    evs.LogTVPGTEvaluator
]

@pytest.mark.parametrize("ev", evaluators)
def test_auto_skip(ev: Evaluator):
    """
    | **Description**
    |   Automatically skips modules that are not tested for
    """
    if ev in [evs.PrintNewDataPointsEvaluator,
            evs.PrintQueryEvaluator,
            evs.PrintExpTimeEvaluator,
            evs.PrintTimeSourceEvaluator,
            evs.PlotNewDataPointsEvaluator,
            evs.PlotAllDataPointsEvaluator,
            evs.PlotQueryDistEvaluator,
            evs.PlotSampledQueriesEvaluator,
            evs.LogOracleEvaluator,
            evs.LogStreamEvaluator,
            evs.LogProcessEvaluator,
            evs.LogResultEvaluator,
            evs.LogAllEvaluator,
            evs.LogTVPGTEvaluator]:
        pytest.skip("Not yet implemented")
    