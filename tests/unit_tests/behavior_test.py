#Test version 2.0 as of ---
from alts.core.oracle.data_behavior import DataBehavior  
import alts.modules.behavior as bem

from alts.modules.data_process.process import DataSourceProcess
from alts.modules.oracle.data_source import TimeBehaviorDataSource
from alts.core.experiment_runner import ExperimentRunner
from alts.modules.query.query_optimizer import NoQueryOptimizer
from alts.modules.query.query_sampler import UniformQuerySampler
from alts.core.query.query_selector import ResultQuerySelector
from alts.core.experiment_modules import InitQueryExperimentModules

import alts.modules.testing_modules as tm
import numpy as np
import pytest

"""
| **Test aims**
|   The data behavior modules are tested for:
|   - Correct output dimensions with respect to paramters
|   - Handling of nonsensical user parameters
"""

behaviors = [
    bem.RandomTimeBrownBehavior
]
behaviors = tm.query_for_members(bem, DataBehavior)
special_behaviors = [
    
]
simple_behaviors = np.setdiff1d(behaviors, special_behaviors, assume_unique=True)

shape_values = [(1,), (1,1)]#, (2,2), (5,3,2), (2,3,4,1)]
change_intervals = [1,5,10,50]
lower_values = [-10,-1,0]
upper_values = [0,1,10]
start_times = [0,10,50]
stop_times = [100,500,600,1000]

@pytest.mark.parametrize("change_interval, lower_value, upper_value, start_time, stop_time", 
                         [(1, -1, 1, 0, 100),
                          (5, 0, 10, 10, 500),
                          (10, -10, 0, 50, 1000),
                          (50, 0, 0, 0, 600)])
@pytest.mark.parametrize("result_shape", shape_values)
@pytest.mark.parametrize("be", simple_behaviors)
def test_basic(be: type[DataBehavior], change_interval, lower_value, upper_value, start_time, stop_time, result_shape):
    pytest.xfail("Fix me")
    bp = tm.TestBlueprint(process=DataSourceProcess(data_source=TimeBehaviorDataSource(result_shape=result_shape, behavior=be(change_interval=change_interval, lower_value=lower_value, upper_value=upper_value, start_time=start_time, stop_time=stop_time))),
                               evaluators=(tm.ConstraintEvaluator(func_path="process.data_source.behavior", q_index=slice(None,None,None), r_index=1),))
    er = ExperimentRunner([bp])
    er.run_experiment(bp)

special_assginments = {

}

@pytest.mark.parametrize("special_behavior", special_behaviors)
def test_special(special_behavior: type[DataBehavior]):
    if (special_behavior not in special_assginments):
        pytest.xfail(f"Special declared DataBehavior {special_behavior.__name__} not tested")
    if special_assginments[special_behavior] is None:
        pytest.xfail(f"Not yet implemented: {special_behavior.__name__}")