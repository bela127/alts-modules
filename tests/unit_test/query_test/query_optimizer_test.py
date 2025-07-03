from alts.core.query.query_optimizer import QueryOptimizer
import alts.modules.query.query_optimizer as qos

from alts.core.experiment import Experiment
from alts.modules.blueprint import BaselineBlueprint
from alts.core.experiment_modules import InitQueryExperimentModules
from alts.modules.query.query_sampler import LatinHypercubeQuerySampler, UniformQuerySampler
from alts.core.query.query_selector import ResultQuerySelector
from alts.modules.query.query_decider import AllQueryDecider
from alts.modules.query.selection_criteria import AllSelectionCriteria

import pytest

"""
| **Test aims**
|   The query optimizers modules are tested for:
|   - Nothing
"""

query_optimizers = [
    qos.NoQueryOptimizer,
    qos.GAQueryOptimizer,
    qos.MaxMCQueryOptimizer,
    qos.ProbWeightedMCQueryOptimizer
]

num_queries = [0,1,3,5,10]

@pytest.mark.parametrize("nq", num_queries)
@pytest.mark.parametrize("qo", query_optimizers)
def test_num_queries(qo: QueryOptimizer, nq: int):
    """
    | **Description**
    |   Tests whether at most num_queries queries have been selected.
    """
    qo = qo()
    exp = Experiment(BaselineBlueprint(experiment_modules= InitQueryExperimentModules(initial_query_sampler=LatinHypercubeQuerySampler(num_queries=nq), 
                                                                                      query_selector=ResultQuerySelector(query_optimizer=qo(selection_criteria=AllSelectionCriteria(), query_sampler=UniformQuerySampler(num_queries=nq)), 
                                                                                                                         query_decider=AllQueryDecider()))),1)
    if type(qo) in {qos.NoQueryOptimizer, qos.MaxMCQueryOptimizer, qos.ProbWeightedMCQueryOptimizer}:
        assert len(exp.experiment_modules.query_selector.query_optimizer.select(num_queries=nq)) == nq, f"QO should select {nq} queries"
    elif type(qo) in {qos.GAQueryOptimizer}:
        assert True, "QO decides number of selected queries on its own"
    else:
        raise ValueError(f"QueryOptimizer not found: {qo}")

@pytest.mark.parametrize("qo", query_optimizers)
def test_auto_skip(qo: QueryOptimizer):
    """
    | **Description**
    |   Automatically skips modules that are not tested for
    """
    if qo in [qos.NoQueryOptimizer,
            qos.GAQueryOptimizer,
            qos.MaxMCQueryOptimizer,
            qos.ProbWeightedMCQueryOptimizer]:
        pytest.skip("NYI")
    