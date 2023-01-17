from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

import numpy as np

from alts.core.query.query_optimizer import QueryOptimizer

from alts.modules.query.selection_criteria import NoSelectionCriteria
from alts.core.configuration import Required, is_set, init, pre_init

if TYPE_CHECKING:
    from typing import Dict, Generator
    from alts.core.query.query_sampler import QuerySampler
    from alts.core.query.selection_criteria import SelectionCriteria
    from alts.core.experiment_modules import ExperimentModules
    from typing_extensions import Self # type: ignore

@dataclass
class NoQueryOptimizer(QueryOptimizer):
    selection_criteria: SelectionCriteria = init(default_factory=NoSelectionCriteria)
    query_sampler: Required[QuerySampler] = init()

    def __post_init__(self):
        super().__post_init__()
        self.query_sampler = is_set(self.query_sampler)(exp_modules = self.exp_modules)


    def select(self):

        queries = is_set(self.query_sampler).sample()
        queries, scores = self.selection_criteria.query(queries)

        return queries, scores


@dataclass
class MCQueryOptimizer(QueryOptimizer):
    query_sampler: QuerySampler  = init()
    num_tries: int = pre_init(default=100)

    def __call__(self, exp_modules: ExperimentModules = None, **kwargs) -> Self:
        obj = super().__call__(exp_modules, **kwargs)
        obj.query_sampler = obj.query_sampler(obj.selection_criteria)
        return obj

    
class MaxMCQueryOptimizer(MCQueryOptimizer):

    def select(self):

        query_candidates = self.query_sampler.sample(self.num_tries)
        query_candidates, candidate_scores = self.selection_criteria.query(query_candidates)

        num_queries = query_candidates.shape[0]

        ind = np.argpartition(candidate_scores, -num_queries)[-num_queries:]
        queries = query_candidates[ind]
        scores = candidate_scores[ind]

        return queries, scores

@dataclass
class ProbWeightedMCQueryOptimizer(MCQueryOptimizer):
    _rng: Generator = pre_init(default_factory = np.random.default_rng)

    def select(self):
        query_candidates = self.query_sampler.sample(self.num_tries)
        scores = self.selection_criteria.query(query_candidates)

        
        scores_zero = np.nan_to_num(scores)
        scores_weight = np.exp(scores_zero)
        score_sum = np.sum(scores_weight)
        weight = scores_weight / score_sum

        num_queries = query_candidates.shape[0]

        indexes = np.arange(num_queries)

        if np.count_nonzero(np.isnan(weight)) > 0:
            idx = self._rng.choice(a=indexes, size=num_queries, replace=False)
        else:
            idx = self._rng.choice(a=indexes, size=num_queries, replace=False, p=weight)

        return query_candidates[idx], scores[idx]