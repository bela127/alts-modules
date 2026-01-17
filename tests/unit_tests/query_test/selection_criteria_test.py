from alts.core.query.selection_criteria import SelectionCriteria
import alts.modules.query.selection_criteria as scm

from alts.core.experiment_modules import InitQueryExperimentModules
from alts.modules.query.query_sampler import UniformQuerySampler
from alts.modules.query.query_decider import AllQueryDecider
from alts.modules.query.query_optimizer import NoQueryOptimizer
from alts.core.query.query_selector import ResultQuerySelector

from alts.modules.data_process.process import DataSourceProcess
from alts.modules.oracle.data_source import RandomUniformDataSource
from alts.core.experiment_runner import ExperimentRunner

import alts.modules.testing_modules as tm
import numpy as np
import pytest

"""
| **Test aims**
|   The data sampler modules are tested for:
|   - Results inside constraints if queries inside constraints
"""

#List of selection criterias
selection_criterias = tm.query_for_members(scm, SelectionCriteria)


shape_values = [(1,), (1,1), (2,2), (5,3,2), (2,3,4,1)]

@pytest.mark.parametrize("query_shape", shape_values)
@pytest.mark.parametrize("sc", selection_criterias)
def test_general(sc: type[SelectionCriteria], query_shape: tuple):
    pytest.skip("Missing Constraints")
    bp = tm.TestBlueprint(process=DataSourceProcess(data_source=RandomUniformDataSource(query_shape=query_shape)),
                          experiment_modules=InitQueryExperimentModules(initial_query_sampler=UniformQuerySampler(num_queries=10), query_selector=ResultQuerySelector(query_optimizer=NoQueryOptimizer(query_sampler=UniformQuerySampler(), selection_criteria=sc), query_decider=AllQueryDecider())),
                          evaluators=(tm.ConstrainEvaluator(func_path="experiment_modules.query_selector.query_optimizer.selection_criteria.query", q_index=slice(None,None,None), r_index=0),))
    er = ExperimentRunner([bp])
    er.run_experiment(bp)