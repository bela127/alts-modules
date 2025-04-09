#Version 1.1.1 conform as of 09.04.2025
"""
| *alts.modules.data_sampler*
| :doc:`Core Module </core/data/data_sampler>`
"""

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
    """
    KDTreeKNNDataSampler(sample_size_max, sample_size_min, sample_size_data_fraction)
    | **Description**
    |   The "k-dimensional tree k-nearest neighbor" data sampler samples the specified number of closest data points to the queries provided.

    :param sample_size_max: Largest allowed sample size (default= 80)
    :type sample_size_max: int
    :param sample_size_min: Smallest allowed sample size (default= 5)
    :type sample_size_min: int
    :param sample_size_data_fraction: Largest (ceiled) allowed sample size relative to total data pool size (default= 6, i.e. 1/6 of data pool)
    :type sample_size_data_fraction: int
    """
    sample_size_max: int = init(default=80)
    sample_size_min: int = init(default=5)
    sample_size_data_fraction: int = init(default=6)

    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Prepares the estimator for the nearest neighbor search.
        """
        super().post_init()
        self._knn = NearestNeighbors(n_neighbors=self.sample_size_max)

    def result_update(self, subscription: Subscribable):
        """
        result_update(self, subscription) -> None
        | **Description**
        |   Fits the nearest neighbor estimator to the updated data.

        :param subscription: The updated subscription
        :type Subscribable: Subscribable
        """
        super().result_update(subscription)
        self._knn.fit(self.data_pools.result.queries, self.data_pools.result.results)

    def query(self, queries, size = None):
        """
        query(self, queries, size) -> data_points
        | **Description**
        |   Samples the ```size``` nearest neighboring data points to the queries provided.

        :param queries: Queries' whose neighborhood to sample from
        :type queries: Tuple[NDArray[Shape["query_nr, ... query_dim"], Number]
        :param size: Number of data points to return (default= max_size)
        :type size: int
        :return: Nearest ```size``` neighbours to the given queries
        :rtype: Tuple[Tuple[NDArray[Shape["query_nr, ... query_dim"], Number], Tuple[NDArray[Shape["result_nr, ... result_dim"], Number]]
        """
        if size is None: 
            size = self.sample_size_max
        if self.data_pools.result.query_constrain().count // self.sample_size_data_fraction < size: # type: ignore
            size = np.ceil(self.data_pools.result.query_constrain().count / self.sample_size_data_fraction) # type: ignore
        if size < self.sample_size_min: 
            size = self.sample_size_min


        kneighbor_indexes = self._knn.kneighbors(queries, n_neighbors=int(size), return_distance=False)

        neighbor_queries = self.data_pools.result.queries[kneighbor_indexes]
        kneighbors = self.data_pools.result.results[kneighbor_indexes]
        
        return (neighbor_queries, kneighbors)
    
    def query_constrain(self):
        """
        query_constrain(self) -> QueryConstrain
        | **Description**
        |   Returns the data sampler's query constraints.

        :return: Query constraints
        :rtype: QueryConstrain
        """
        query_shape = self.data_pools.result.query_constrain().shape
        query_ranges = self.data_pools.result.query_constrain().ranges
        query_count = self.data_pools.result.query_constrain().count
        query_constrain = QueryConstrain(query_count,query_shape,query_ranges)
        
        queries = self.data_pools.result.queries
        query_constrain.ranges = queries[:,None]
        query_constrain._last_queries = self.data_pools.result.last_queries

        return query_constrain

    def result_constrain(self):
        """
        result_constrain(self) -> ResultConstrain
        | **Description**
        |   Returns the data sampler's result constraints.

        :return: Result constraints
        :rtype: ResultConstrain
        """
        return self.data_pools.result.result_constrain()

    
@dataclass
class KDTreeRegionDataSampler(ResultDataSampler):
    """
    KDTreeRegionDataSampler(region_size)
    | **Description**
    |   The "k-dimensional region" data sampler samples all data points in the given radius around the given queries.

    :param region_size: Radius around given queries whose data points are sampled
    :type region_size: float
    """
    region_size: float = init(default=0.1)

    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Prepares the estimator for the nearest neighbor search.
        """
        super().post_init()
        self._knn = NearestNeighbors()
        

    def result_update(self, subscription: Subscribable):
        """
        result_update(self, subscription) -> None
        | **Description**
        |   Fits the nearest neighbor estimator to the updated data.

        :param subscription: The updated subscription
        :type Subscribable: Subscribable
        """
        super().result_update(subscription)
        self._knn.fit(self.data_pools.result.queries, self.data_pools.result.results)

    def query(self, queries, size = None):
        """
        query(self, queries, size) -> data_points
        | **Description**
        |   Samples all neighboring data points in ```region_size``` radius to the queries provided.

        :param queries: Queries' whose neighborhood to sample from
        :type queries: Tuple[NDArray[Shape["query_nr, ... query_dim"], Number]
        :param size: Number of data points to return (ignored)
        :type size: int
        :return: All neighbours in ```region_size``` radius to the given queries
        :rtype: Tuple[Tuple[NDArray[Shape["query_nr, ... query_dim"], Number], Tuple[NDArray[Shape["result_nr, ... result_dim"], Number]]
        """
        kneighbor_indexes = self._knn.radius_neighbors(queries, radius=self.region_size ,return_distance=False)

        neighbor_queries = np.asarray([self.data_pools.result.queries[kneighbor_indexe] for kneighbor_indexe in kneighbor_indexes], dtype=object)
        kneighbors = np.asarray([self.data_pools.result.results[kneighbor_indexe] for kneighbor_indexe in kneighbor_indexes], dtype=object)
        
        return (neighbor_queries, kneighbors)
    
    def query_constrain(self):
        """
        query_constrain(self) -> QueryConstrain
        | **Description**
        |   Returns the data sampler's query constraints.

        :return: Query constraints
        :rtype: QueryConstrain
        """
        query_shape = self.data_pools.result.query_constrain().shape
        query_ranges = self.data_pools.result.query_constrain().ranges
        query_count = self.data_pools.result.query_constrain().count
        query_constrain = QueryConstrain(count=query_count,shape=query_shape,ranges=query_ranges)
        
        queries = self.data_pools.result.query_constrain().all_queries()
        query_constrain.ranges = queries[:,None]
        query_constrain._last_queries = self.data_pools.result.query_constrain().last_queries()

        return query_constrain

    def result_constrain(self):
        """
        result_constrain(self) -> ResultConstrain
        | **Description**
        |   Returns the data sampler's result constraints.

        :return: Result constraints
        :rtype: ResultConstrain
        """
        return self.data_pools.result.result_constrain()
