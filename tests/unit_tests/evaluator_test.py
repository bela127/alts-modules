from alts.core.evaluator import Evaluator
import alts.modules.evaluator as evm

from alts.core.experiment_runner import ExperimentRunner

import alts.modules.testing_modules as tm
import numpy as np
import pytest


"""
| **Test aims**
|   The evaluator modules are tested for:
|   - Not causing any errors
"""

evaluators = tm.query_for_members(evm, Evaluator)
special_evaluators = [
    evm.PrintNewDataPointsEvaluator, #ResultDataPools
    evm.PrintQueryEvaluator,  #POracles
    evm.PlotNewDataPointsEvaluator, #ResultDataPools
    evm.PlotAllDataPointsEvaluator, #ResultDataPools
    evm.PlotQueryDistEvaluator, #POracles
    evm.LogOracleEvaluator, #POracles
    evm.LogStreamEvaluator, #StreamDataPools
    evm.LogProcessEvaluator, #ProcessDataPools
    evm.LogResultEvaluator, #ResultDataPools
    evm.LogAllEvaluator, #Stream-/Process- and Result- DataPools
    evm.LogTVPGTEvaluator, #DelayedProcess
]
simple_evaluators = np.setdiff1d(evaluators, special_evaluators, assume_unique=True)

@pytest.mark.parametrize("ev", simple_evaluators)
def test_basic(ev: type[Evaluator]):
    bp = tm.TestBlueprint(evaluators=(ev(),))
    er = ExperimentRunner([bp])
    er.run_experiment(bp)
    
special_assginments = {
    evm.PrintNewDataPointsEvaluator: None, #ResultDataPools
    evm.PrintQueryEvaluator: None,  #POracles
    evm.PlotNewDataPointsEvaluator: None, #ResultDataPools
    evm.PlotAllDataPointsEvaluator: None, #ResultDataPools
    evm.PlotQueryDistEvaluator: None, #POracles
    evm.LogOracleEvaluator: None, #POracles
    evm.LogStreamEvaluator: None, #StreamDataPools
    evm.LogProcessEvaluator: None, #ProcessDataPools
    evm.LogResultEvaluator: None, #ResultDataPools
    evm.LogAllEvaluator: None, #Stream-/Process- and Result- DataPools
    evm.LogTVPGTEvaluator: None, #DelayedProcess
}

@pytest.mark.parametrize("special_evaluator", special_evaluators)
def test_special_tested(special_evaluator: type[Evaluator]):
    if (special_evaluator not in special_assginments):
        pytest.xfail(f"Special declared Evaluator {special_evaluator.__name__} not tested")
    if special_assginments[special_evaluator] is None:
        pytest.xfail("Not yet implemented")