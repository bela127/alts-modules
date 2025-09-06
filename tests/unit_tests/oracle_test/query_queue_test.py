#Test version 2.0 as of 06.09.2025
from alts.core.oracle.query_queue import QueryQueue
import alts.modules.oracle.query_queue as qq

from alts.modules.data_process.process import DataSourceProcess
from alts.modules.oracle.data_source import RandomUniformDataSource
from alts.core.experiment_runner import ExperimentRunner

import alts.modules.testing_modules as tm
import numpy as np
import pytest

"""
| **Test aims**
|   The query queue modules are tested for:
|   - Correct output dimensions with respect to paramters
"""

query_queues = [
    qq.FCFSQueryQueue
]

shape_values = [(1,), (1,1), (2,2), (5,3,2), (2,3,4,1)]

@pytest.mark.parametrize("query_shape", shape_values)
def test_FCFSQueryQueue(query_shape: tuple):
    bp = tm.TestBlueprint(process=DataSourceProcess(data_source=RandomUniformDataSource(query_shape=query_shape)),
                               evaluators=(tm.ConstrainEvaluator(func_path="oracles.process.pop", r_index=slice(None,None,None)),))
    er = ExperimentRunner([bp])
    er.run_experiment(bp)

