#Test version 2.0 as of ---
from alts.core.oracle.data_behavior import DataBehavior  
import alts.modules.behavior as beh

from alts.modules.data_process.process import DataSourceProcess
from alts.modules.oracle.data_source import RandomUniformDataSource
from alts.core.experiment_runner import ExperimentRunner

import alts.modules.testing_modules as tm
import numpy as np
import pytest

"""
| **Test aims**
|   The data behavior modules are tested for:
|   - Inside result constraints given queries inside query constraints
"""

behaviors = [
    beh.EquidistantTimeUniformBehavior,
    beh.RandomTimeUniformBehavior,
    beh.RandomTimeBrownBehavior
]


shape_values = [(1,), (1,1), (2,2), (5,3,2), (2,3,4,1)]

@pytest.mark.parametrize("query_shape", shape_values)
def test_behavior(query_shape: tuple):
    pytest.skip("WIP")
    bp = tm.TestBlueprint(process=DataSourceProcess(data_source=RandomUniformDataSource(query_shape=query_shape)),
                               evaluators=(tm.ConstrainEvaluator(func_path="oracles.process.pop", r_index=slice(None,None,None)),))
    er = ExperimentRunner([bp])