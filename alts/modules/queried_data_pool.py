#Version 1.1.1 conform as of 15.04.2025
"""
| *alts.modules.queried_data_pool*
| :doc:`Core Module </core/data/queried_data_pool>`
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from random import choice

import numpy as np
from alts.core.data.constrains import QueryConstrain, ResultConstrain

from alts.core.data.queried_data_pool import QueriedDataPool

if TYPE_CHECKING:
    from typing import Dict

class FlatQueriedDataPool(QueriedDataPool):
    """
    FlatQueriedDataPool()
    | **Description**
    |   This queryable data pool returns a random result matching the query.
    """

    def __init__(self):
        """
        __init__(self) -> None
        | **Description**
        |   Initializes empty lists and dicts.
        """
        super().init(FlatQueriedDataPool)
        self.query_index: Dict = {}

    
    def query(self, queries):
        """
        query(self, queries) -> data_points
        | **Description**
        |   Returns a list containing a random associated result for each query in queries.

        :param queries: Requested queries
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: The tuple of queries and their associated results
        :rtype: Tuple[`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_,`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_]
        """
        result_list = []
        for query in queries:
            result_candidate = self.query_index.get(tuple(query), [])
            result = choice(result_candidate)
            result_list.append(result)
        
        results: np.ndarray = np.asarray(result_list)
        return queries, results


    def add(self, data_points):
        """
        add(self, data_points) -> None
        | **Description**
        |   Adds the new result to the query to the already existing results to the same query.

        :param data_points: A tuple of queries and their associated results. 
        :type data_points: Tuple[`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_,`NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_]
        """
        queries, results = data_points
        for query, result in zip(queries, results):

            results = self.query_index.get(tuple(query), [])
            self.query_index[tuple(query)] = results + [result]
        super().add(data_points)

    def query_constrain(self) -> QueryConstrain:
        """
        query_constrain(self) -> QueryConstrain
        | **Description**
        |   Returns its own query constraints.

        :return: Data pool's query constraints
        :rtype: QueryConstrain
        """
        return QueryConstrain(count = self.queries.shape[0], shape = self._query_constrain().shape, ranges = self.queries)


    def result_constrain(self) -> ResultConstrain:
        """
        result_constrain(self) -> ResultConstrain
        | **Description**
        |   Returns its own result constraints.

        :return: Data pool's result constraints
        :rtype: ResultConstrain
        """
        return self._result_constrain()