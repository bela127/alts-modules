#Version 1.1 conform as of 05.10.2024
"""
:doc:`Core Module </core/query/query_decider>`
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field
from abc import abstractmethod

import numpy as np

from alts.core.configuration import Configurable, init, Required, is_set
from alts.core.query.query_decider import QueryDecider

if TYPE_CHECKING:
    from typing import Tuple, Optional
    from nptyping import NDArray, Number, Shape

@dataclass    
class AllQueryDecider(QueryDecider):
    """
    AllQueryDecider()
    | **Description**
    |   Always picks all query candidates.
    """
    def decide(self, query_candidates: NDArray[Shape["query_nr, ... query_dims"], Number], scores: NDArray[Shape["query_nr, [query_score]"], Number]) -> Tuple[bool, NDArray[Shape["query_nr, ... query_dims"], Number]]: # type: ignore
        """
        decide(self, query_candidates, scores) -> (bool, queries)
        | **Description**
        |   Returns the list of candidates. 
        |   Decides always.

        :param query_candidates: A list of queries to choose from
        :type query_candidates: Iterable over `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_ 
        :param scores: A list of scores associated to the queries in ``query_candidates`` (Not used here)
        :type scores: Iterable over `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_ 
        :return: True, query_candidates
        :rtype: boolean, Iterable over `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_ 
        """
        return True, query_candidates

@dataclass    
class TopKQueryDecider(QueryDecider):
    """
    TopKQueryDecider(k)
    | **Description**
    |   Always picks the ``k`` best query candidates based on their score.

    :param k: How many queries to pick (default = 4)
    :type k: int
    """
    k: int = init(default= 4)

    def decide(self, query_candidates: NDArray[Shape["query_nr, ... query_dims"], Number], scores: NDArray[Shape["query_nr, [query_score]"], Number]) -> Tuple[bool, NDArray[Shape["query_nr, ... query_dims"], Number]]: # type: ignore
        """
        decide(self, query_candidates, scores) -> (bool, queries)
        | **Description**
        |   Returns the ``k`` query candidates with the highest scores.
        |   Decides always.

        :param query_candidates: A list of queries to choose from
        :type query_candidates: Iterable over `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_ 
        :param scores: A list of scores associated to the queries in ``query_candidates`` (Not used here)
        :type scores: Iterable over `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_ 
        :return: True, query_candidates
        :rtype: boolean, Iterable over `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_ 
        """
        if query_candidates.size > self.k:
            ind = np.argpartition(scores, -self.k, axis=0)[-self.k:]
            queries = query_candidates[ind]
        else:
            queries = query_candidates
        return True, queries

@dataclass
class NoQueryDecider(QueryDecider):
    """
    NoQueryDecider()
    | **Description**
    |   Never picks any query candidates.
    """
    def decide(self, query_candidates: NDArray[Shape["query_nr, ... query_dims"], Number], scores: NDArray[Shape["query_nr, [query_score]"], Number]) -> Tuple[bool, NDArray[Shape["query_nr, ... query_dims"], Number]]: # type: ignore
        """
        decide(self, query_candidates, scores) -> (bool, queries)
        | **Description**
        |   Returns an empty query_candidates list. 
        |   Decides never.

        :param query_candidates: A list of queries to choose from
        :type query_candidates: Iterable over `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_ 
        :param scores: A list of scores associated to the queries in ``query_candidates`` (Not used here)
        :type scores: Iterable over `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_ 
        :return: False, Empty list of queries
        :rtype: boolean, Iterable over `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_ 
        """
        query = np.empty((0, *query_candidates.shape[1:]))
        return False, query

@dataclass
class ThresholdQueryDecider(QueryDecider):
    """
    ThresholdQueryDecider(threshold)
    | **Description**
    |   Picks all query candidates that have a score above a certain threshold.

    :param threshold: The threshold the query candidates have to exceed (defaults = 0.5)
    :type k: float
    """
    threshold: float = init(default= 0.5)

    def decide(self, query_candidates: NDArray[Shape["query_nr, ... query_dims"], Number], scores: NDArray[Shape["query_nr, [query_score]"], Number]) -> Tuple[bool, NDArray[Shape["query_nr, ... query_dims"], Number]]: # type: ignore
        """
        decide(self, query_candidates, scores) -> (bool, queries)
        | **Description**
        |   Returns all query candidates with a score above the threshold. 
        |   Decides only if there are any query candidates fulfilling this requirement.

        :param query_candidates: A list of queries to choose from
        :type query_candidates: Iterable over `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_ 
        :param scores: A list of scores associated to the queries in ``query_candidates`` (Not used here)
        :type scores: Iterable over `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_ 
        :return: Whether it wants to decide, List of query candidates above the threshold
        :rtype: boolean, Iterable over `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_ 
        """
        query = query_candidates[scores[:,0]>self.threshold]
        flag = query.shape[0] > 0
        return flag, query
