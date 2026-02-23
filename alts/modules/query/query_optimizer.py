#Version 1.1.1 conform as of 23.04.2025
"""
| *alts.modules.query.query_optimizer*
| :doc:`Core Module </core/query/query_optimizer>`
"""
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
    """
    NoQueryOptimizer(selection_criteria, query_sampler)
    | **Description**
    |   Selects the first queries from the query sample.

    :param selection_criteria: Scores the queries for the optimizer
    :type selection_criteria: SelectionCriteria
    :param query_sampler: Samples queries to work with
    :type query_sampler: QuerySampler
    """
    selection_criteria: SelectionCriteria = init(default_factory=NoSelectionCriteria)
    query_sampler: QuerySampler = init()

    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Initializes the query_sampler
        """
        super().post_init()
        self.query_sampler = self.query_sampler(exp_modules=self.exp_modules)


    def select(self, num_queries = None):
        """
        select(self) -> queries, scores
        | **Description**
        |   Selects the first sampled queries regardless of score

        :return: queries and associated scores scores
        :rtype: queries, `NDArray[float] <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        """
        queries = self.query_sampler.sample(num_queries)
        queries, scores = self.selection_criteria.query(queries)

        return queries, scores
    

@dataclass
class GAQueryOptimizer(QueryOptimizer):
    """
    GAQueryOptimizer()
    | **Description**
    |   The Genetic Algortihm Query Optimizer tries to maximize the query scores through Differential Evolution
    """

    def select(self):
        """
        select(self) -> queries, scores
        | **Description**
        |   Tries to find the score maximizing queries through heuristic methods.

        :return: queries, scores
        :rtype: `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_, `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        """
        def opt_func(x):
            """
            opt_func(queries's) -> scores
            | **Description**
            |   Returns the scores to all given queries

            :param x: queries's
            :type x: Iterable over iterables over `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
            :return: Scores of all queries
            :rtype: Iterable over `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
            """
            queries = x[:,None]
            queries, scores = self.selection_criteria.query(queries)
            return scores[0]
        res = differential_evolution(opt_func, bounds=np.repeat(self.oracles.query_constrain().ranges, 2, axis=0)) # type: ignore
        queries = res.x[:,None]
        queries, scores = self.selection_criteria.query(queries)
        
        return queries, scores


@dataclass
class MCQueryOptimizer(QueryOptimizer):
    """
    MCQueryOptimizer(query_sampler, num_tries=100)
    | **Description**
    |   The Monte Carlo Query Optimizer works by sampling ``num_tries`` many times and choosing one of those (asbtract class).

    :param query_sampler: The query sampler to use
    :type query_sampler: QuerySampler
    :param num_tries: Amount of samples to get (default=100)
    :type num_tries: int
    """
    query_sampler: QuerySampler  = init()
    num_tries: int = init(default=100)

    def post_init(self):
        """ 
        post_init(self) -> None
        | **Description**
        |   Initializes the query sampler
        """
        super().post_init()
        self.query_sampler = self.query_sampler(exp_modules=self.exp_modules, query_constrain=self.query_constrain)

@dataclass
class MaxMCQueryOptimizer(MCQueryOptimizer):
    """
    MaxMCQueryOptimizer(query_sampler, num_tries)
    | **Description**
    |   The Maximizing Monte Carlo Query Optimizer samples ``num_tries`` many times and then chooses the best queries.

    :param query_sampler: The query sampler to use
    :type query_sampler: QuerySampler
    :param num_tries: Amount of samples to get (default= 100)
    :type num_tries: int
    """

    def select(self):
        """
        select(self) -> queries, scores
        | **Description**
        |   Selects the ```num_queries``` queries among ```num_tries``` samples with the highest score.

        :return: Queries, scores
        :rtype: `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_, `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        """
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
    """
    ProbWeightedMCQueryOptimizer(query_sampler, num_tries)
    | **Description**
    |   The Probabilistic Weighted Monte Carlo Query Optimizer selects samples randomly with highly scored queries having a higher probability.

    :param query_sampler: The query sampler to use
    :type query_sampler: QuerySampler
    :param num_tries: Amount of samples to get (default= 100)
    :type num_tries: int
    """
    _rng: Generator = pre_init(default_factory = np.random.default_rng)

    def select(self):
        """
        select(self) -> queries, scores
        | **Description**
        |   Gives each query a relative weight exponential to their score.
        |   If the value of any weight is NaN, selects ```num_queries``` queries randomly with equal probability distribution.
        |   Otherwise, selects ```num_queries``` queries randomly with a probability distribution equal to their weights.

        :return: Queries, scores
        :rtype: `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_, `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        """
        query_candidates = self.query_sampler.sample(self.num_tries)
        scores = self.selection_criteria.query(query_candidates)

        
        scores_zero = np.nan_to_num(scores)
        scores_weight = np.exp(scores_zero)
        score_sum = np.sum(scores_weight)
        weight = scores_weight / score_sum

        num_queries = self.query_sampler.num_queries

        indexes = np.arange(num_queries)

        if np.count_nonzero(np.isnan(weight)) > 0:
            idx = self._rng.choice(a=indexes, size=num_queries, replace=False) # type: ignore
        else:
            idx = self._rng.choice(a=indexes, size=num_queries, replace=False, p=weight) # type: ignore

        return query_candidates[idx], scores[idx]