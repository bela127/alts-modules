from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

import numpy as np
from sklearn.neighbors import NearestNeighbors

from alts.core.data.data_sampler import ResultDataSampler
from alts.core.data.constrains import QueryConstrain, ResultConstrain
from alts.core.configuration import init

if TYPE_CHECKING:
    from typing import Tuple
    from alts.core.subscribable import Subscribable

@dataclass
class KDTreeKNNDataSampler(ResultDataSampler):
    sample_size: int = init(default=50)
    sample_size_data_fraction: int = init(default=6)

    def post_init(self):
        super().post_init()
        self._knn = NearestNeighbors(n_neighbors=self.sample_size)

    def result_update(self, subscription: Subscribable):
        super().result_update(subscription)
        self._knn.fit(self.data_pools.result.queries, self.data_pools.result.results)

    def query(self, queries, size = None):
        if size is None: size = self.sample_size
        if self.data_pools.result.query_constrain().count // self.sample_size_data_fraction < size: size = np.ceil(self.data_pools.result.query_constrain().count / self.sample_size_data_fraction)

        kneighbor_indexes = self._knn.kneighbors(queries, n_neighbors=int(size), return_distance=False)

        neighbor_queries = self.data_pools.result.queries[kneighbor_indexes]
        kneighbors = self.data_pools.result.results[kneighbor_indexes]
        
        return (neighbor_queries, kneighbors)
    
    def query_constrain(self):
        query_shape = self.data_pools.result.query_constrain().shape
        query_ranges = self.data_pools.result.query_constrain().ranges
        query_count = self.data_pools.result.query_constrain().count
        query_constrain = QueryConstrain(count=query_count,shape=query_shape,ranges=query_ranges)
        
        queries = self.data_pools.result.query_constrain().all_queries()
        query_constrain.ranges = queries[:,None]
        query_constrain._last_queries = self.data_pools.result.query_constrain().last_queries()

        return query_constrain

    def result_constrain(self):
        return self.data_pools.result.result_constrain()

    
@dataclass
class KDTreeRegionDataSampler(ResultDataSampler):
    region_size: float = init(default=0.1)

    def post_init(self):
        super().post_init()
        self._knn = NearestNeighbors()
        

    def result_update(self, subscription: Subscribable):
        super().result_update(subscription)
        self._knn.fit(self.data_pools.result.queries, self.data_pools.result.results)

    def query(self, queries, size = None):

        kneighbor_indexes = self._knn.radius_neighbors(queries, radius=self.region_size ,return_distance=False)

        neighbor_queries = np.asarray([self.data_pools.result.queries[kneighbor_indexe] for kneighbor_indexe in kneighbor_indexes], dtype=object)
        kneighbors = np.asarray([self.data_pools.result.results[kneighbor_indexe] for kneighbor_indexe in kneighbor_indexes], dtype=object)
        
        return (neighbor_queries, kneighbors)
    
    def query_constrain(self):
        query_shape = self.data_pools.result.query_constrain().shape
        query_ranges = self.data_pools.result.query_constrain().ranges
        query_count = self.data_pools.result.query_constrain().count
        query_constrain = QueryConstrain(count=query_count,shape=query_shape,ranges=query_ranges)
        
        queries = self.data_pools.result.query_constrain().all_queries()
        query_constrain.ranges = queries[:,None]
        query_constrain._last_queries = self.data_pools.result.query_constrain().last_queries()

        return query_constrain

    def result_constrain(self):
        return self.data_pools.result.result_constrain()
