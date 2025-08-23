from alts.core.experiment import Experiment
from alts.modules.blueprint import BaselineBlueprint
from alts.core.experiment_modules import InitQueryExperimentModules
from alts.modules.query.query_sampler import LatinHypercubeQuerySampler, UniformQuerySampler
from alts.core.query.query_selector import ResultQuerySelector
from alts.modules.query.query_decider import AllQueryDecider
from alts.modules.query.selection_criteria import AllSelectionCriteria

import alts.modules.query.query_optimizer as qos

qo = qos.MaxMCQueryOptimizer
nq = 5
exp = Experiment(BaselineBlueprint(experiment_modules= InitQueryExperimentModules(initial_query_sampler=LatinHypercubeQuerySampler(num_queries=nq), 
                                                                                      query_selector=ResultQuerySelector(query_optimizer=qo(selection_criteria=AllSelectionCriteria(), query_sampler=UniformQuerySampler(num_queries=nq)), 
                                                                                                                         query_decider=AllQueryDecider()))),1)