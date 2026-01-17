from alts.core.data_process.process import Process
import alts.modules.data_process.process as pr

from alts.modules.data_process.process import DataSourceProcess
from alts.modules.oracle.data_source import RandomUniformDataSource
from alts.core.experiment_runner import ExperimentRunner
from alts.modules.query.query_optimizer import NoQueryOptimizer
from alts.modules.query.query_sampler import UniformQuerySampler
from alts.core.query.query_selector import ResultQuerySelector
from alts.core.experiment_modules import InitQueryExperimentModules

import numpy as np
import alts.modules.testing_modules as tm
import pytest

"""
| **Test aims**
|   The processes modules are tested for:
|   - Correct output dimensions with respect to paramters
|   - Handling of nonsensical user parameters
"""

processes = tm.query_for_members(pr, Process)
special_processes = [
    pr.StreamProcess,
    pr.DataSourceProcess,
    pr.DelayedProcess,
    pr.DelayedStreamProcess,
    pr.IntegratingDSProcess, 
    pr.WindowDSProcess
]
simple_processes = np.setdiff1d(processes, special_processes, assume_unique=True)


@pytest.mark.parametrize("pr", simple_processes)
def test_basic(pr: type[Process], query_shape: tuple):
    bp = tm.TestBlueprint(process=pr(),
                               evaluators=(tm.ConstrainEvaluator(func_path="experiment_modules.query_selector.query_decider.decide", q_index=slice(None,None,None), r_index=1),))
    er = ExperimentRunner([bp])
    er.run_experiment(bp)

special_assginments = {
    pr.StreamProcess: None,
    pr.DataSourceProcess: None,
    pr.DelayedProcess: None,
    pr.DelayedStreamProcess: None,
    pr.IntegratingDSProcess: None, 
    pr.WindowDSProcess: None
}

@pytest.mark.parametrize("special_process", special_processes)
def test_special_tested(special_process: type[Process]):
    assert special_process in special_assginments, f"Special declared Process {special_process.__name__} not tested"
    if special_assginments[special_process] is None:
        pytest.skip("Not yet implemented")
    