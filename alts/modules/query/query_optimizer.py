from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import differential_evolution

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
    query_sampler: QuerySampler = init()

    def post_init(self):
        super().post_init()
        self.query_sampler = self.query_sampler(exp_modules = self.exp_modules)


    def select(self):

        queries = self.query_sampler.sample()
        queries, scores = self.selection_criteria.query(queries)

        return queries, scores
    

@dataclass
class GAQueryOptimizer(QueryOptimizer):
    #QUESTION what does it do
    def select(self):

        def opt_func(x):
            queries = x[:,None]
            queries, scores = self.selection_criteria.query(queries)
            return scores[0]
        res = differential_evolution(opt_func, bounds=np.repeat(self.oracles.query_constrain().ranges, 2, axis=0))
        queries = res.x[:,None]
        queries, scores = self.selection_criteria.query(queries)
        
        return queries, scores


@dataclass
class MCQueryOptimizer(QueryOptimizer):
    query_sampler: QuerySampler  = init()
    num_tries: int = init(default=100)

    def post_init(self):
        super().post_init()
        self.query_sampler = self.query_sampler(exp_modules=self.exp_modules)

@dataclass
class MaxMCQueryOptimizer(MCQueryOptimizer):

    def select(self):

        query_candidates = self.query_sampler.sample(self.num_tries)
        query_candidates, candidate_scores = self.selection_criteria.query(query_candidates)

        num_queries = self.query_sampler.num_queries

        ind = np.argpartition(candidate_scores, -num_queries, axis=0)[-num_queries:]
        ind = ind[:, 0]
        queries = query_candidates[ind, ...]
        scores = candidate_scores[ind, ...]

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

        num_queries = self.query_sampler.num_queries

        indexes = np.arange(num_queries)

        if np.count_nonzero(np.isnan(weight)) > 0:
            idx = self._rng.choice(a=indexes, size=num_queries, replace=False)
        else:
            idx = self._rng.choice(a=indexes, size=num_queries, replace=False, p=weight)

        return query_candidates[idx], scores[idx]