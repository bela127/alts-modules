from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

import numpy as np
from sklearn.neighbors import NearestNeighbors
from alts.core.data.data_pool import DataPool

from alts.core.data_sampler import DataSampler
from alts.core.query.query_pool import QueryPool

if TYPE_CHECKING:
    from typing import Tuple

class KDTreeKNNDataSampler(DataSampler):
    sample_size: int = 50
    sample_size_data_fraction: int = 6


    def __init__(self, sample_size, sample_size_data_fraction = 6):
        super().__init__()
        self.sample_size = sample_size
        self.sample_size_data_fraction = sample_size_data_fraction
        self._knn = NearestNeighbors(n_neighbors=self.sample_size)
        

    def update(self):
        self._knn.fit(self.exp_modules.queried_data_pool.queries, self.exp_modules.queried_data_pool.results)

    def query(self, queries, size = None):
        if size is None: size = self.sample_size
        if self.exp_modules.queried_data_pool.query_pool.query_count // self.sample_size_data_fraction < size: size = np.ceil(self.exp_modules.queried_data_pool.query_pool.query_count / self.sample_size_data_fraction)

        kneighbor_indexes = self._knn.kneighbors(queries, n_neighbors=int(size), return_distance=False)

        neighbor_queries = self.exp_modules.queried_data_pool.queries[kneighbor_indexes]
        kneighbors = self.exp_modules.queried_data_pool.results[kneighbor_indexes]
        
        return (neighbor_queries, kneighbors)
    
    @property
    def query_pool(self):
        query_shape = self.exp_modules.oracle_data_pool.query_shape
        query_ranges = self.exp_modules.oracle_data_pool.query_ranges
        query_count = self.exp_modules.queried_data_pool.query_pool.query_count
        query_pool = QueryPool(query_count=query_count,query_shape=query_shape,query_ranges=query_ranges)
        
        queries = self.exp_modules.queried_data_pool.query_pool.all_queries()
        query_pool._queries = queries
        query_pool._last_queries = self.exp_modules.queried_data_pool.query_pool.last_queries()

        return query_pool

    @property
    def data_pool(self):
        return DataPool(self.query_pool, self.exp_modules.oracle_data_pool.result_shape)

    

class KDTreeRegionDataSampler(DataSampler):

    def __init__(self, region_size = 0.1):
        super().__init__()
        self.region_size = region_size
        self._knn = NearestNeighbors()
        

    def update(self):
        self._knn.fit(self.exp_modules.queried_data_pool.queries, self.exp_modules.queried_data_pool.results)

    def query(self, queries, size = None):

        kneighbor_indexes = self._knn.radius_neighbors(queries, radius=self.region_size ,return_distance=False)

        neighbor_queries = np.asarray([self.exp_modules.queried_data_pool.queries[kneighbor_indexe] for kneighbor_indexe in kneighbor_indexes], dtype=object)
        kneighbors = np.asarray([self.exp_modules.queried_data_pool.results[kneighbor_indexe] for kneighbor_indexe in kneighbor_indexes], dtype=object)
        
        return (neighbor_queries, kneighbors)
    
    @property
    def query_pool(self):
        query_shape = self.exp_modules.oracle_data_pool.query_shape
        query_ranges = self.exp_modules.oracle_data_pool.query_ranges
        query_count = self.exp_modules.queried_data_pool.query_pool.query_count
        query_pool = QueryPool(query_count=query_count,query_shape=query_shape,query_ranges=query_ranges)
        
        queries = self.exp_modules.queried_data_pool.query_pool.all_queries()
        query_pool._queries = queries
        query_pool._last_queries = self.exp_modules.queried_data_pool.query_pool.last_queries()

        return query_pool

    @property
    def data_pool(self):
        return DataPool(self.query_pool, self.exp_modules.oracle_data_pool.result_shape)
