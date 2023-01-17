from __future__ import annotations
from typing import TYPE_CHECKING

from random import choice

import numpy as np
from alts.core.data.constrains import QueryConstrain, ResultConstrain

from alts.core.data.queried_data_pool import QueriedDataPool
from alts.core.query.query_pool import QueryPool

if TYPE_CHECKING:
    from typing import Dict

class FlatQueriedDataPool(QueriedDataPool):
    """
    implements a pool of already labeled data
    """

    iteration = 0

    def __init__(self):
        super().__init__()
        self.query_index: Dict = {}

    
    def query(self, queries):
        result_list = []
        for query in queries:
            result_candidate = self.query_index.get(tuple(query), [])
            result = choice(result_candidate)
            result_list.append(result)
        
        results: np.ndarray = np.asarray(result_list)
        return queries, results


    def add(self, data_points):
        super().add(data_points)

        queries, results = data_points

        for query, result in zip(queries, results):

            results = self.query_index.get(tuple(query), [])
            self.query_index[tuple(query)] = results + [result]

    @property
    def query_constrain(self) -> QueryConstrain:
        return QueryConstrain(count = self.queries.shape[0], query_shape = self._query_constrain.shape, query_ranges = self.queries)


    @property
    def result_constrain(self) -> ResultConstrain:
        return ResultConstrain(shape=self._result_constrain.shape)