from alts.core.evaluator import Evaluator
import alts.modules.evaluator as evm

from alts.core.experiment_runner import ExperimentRunner
from alts.core.data.data_pools import StreamDataPools

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
    #Requires StreamDataPools
    evm.LogStreamEvaluator,
    #Requires ProcessDataPools
    evm.LogProcessEvaluator,
    #Requires DelayedProcess
    evm.LogTVPGTEvaluator, 
    #Requires StreamDataPools, ProcessDataPools, ResultDataPools
    evm.LogAllEvaluator,
]
simple_evaluators = np.setdiff1d(evaluators, special_evaluators, assume_unique=True)

@pytest.mark.parametrize("ev", simple_evaluators)
def test_evaluator(ev: type[Evaluator]):
    bp = tm.TestBlueprint(evaluators=(ev(),))
    er = ExperimentRunner([bp])
    er.run_experiment(bp)

special_assginments = {

}

@pytest.mark.parametrize("special_evaluator", special_evaluators)
def test_special_tested(special_evaluator: type[Evaluator]):
    if (special_evaluator not in special_assginments):
        pytest.xfail(f"Special declared Evaluator {special_evaluator.__name__} not tested")
    if special_assginments[special_evaluator] is None:
        pytest.xfail("Not yet implemented")