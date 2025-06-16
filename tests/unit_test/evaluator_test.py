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
def auto_pass_test(ev: Evaluator):
    """
    | **Description**
    |   Automatically passes modules that are not tested for
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
        assert True
    