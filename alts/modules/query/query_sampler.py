from __future__ import annotations
from typing import TYPE_CHECKING

from math import ceil
import random
from dataclasses import dataclass
from alts.core.data.data_pools import ResultDataPools, StreamDataPools, ProcessDataPools
from alts.core.oracle.oracles import POracles


import numpy as np
from scipy.stats import qmc # type: ignore

from alts.core.query.query_sampler import QuerySampler
from alts.core.configuration import init
from alts.core.data.queried_data_pool import QueriedDataPool

if TYPE_CHECKING:
    from typing import Tuple, List, Union, Literal
    from nptyping import NDArray, Number, Shape

@dataclass
class OptimalQuerySampler(QuerySampler):
    optimal_queries: Tuple[NDArray[Shape["query_nr, ... query_dims"], Number], ...] = init() # type: ignore

    def post_init(self):
        super().post_init()
        for optimal_query in self.optimal_queries:
            if not self.oracles.query_constrain().constrains_met(optimal_query):
                raise ValueError("optimal_queries do not meet oracles.query_constrain")

    def sample(self, num_queries = None):
        if num_queries is None: num_queries = self.num_queries
        
        query_nr = self.optimal_queries[0].shape[0]
        k = ceil(num_queries / query_nr) 
        queries = random.choices(self.optimal_queries, k=k)
        queries = np.concatenate(queries)
        return queries[:num_queries]

@dataclass
class FixedQuerySampler(QuerySampler):
    fixed_query: NDArray[Shape["... query_dims"], Number] = init() # type: ignore

    def post_init(self):
        super().post_init()
        if not self.oracles.query_constrain().constrains_met(self.fixed_query):
            raise ValueError("fixed_query does not meet oracles.query_constrain")

    def sample(self, num_queries = None):
        if num_queries is None: num_queries = self.num_queries
        
        queries = np.repeat(self.fixed_query[None, ...], num_queries, axis=0)
        return queries
@dataclass
class UniformQuerySampler(QuerySampler):

    def sample(self, num_queries = None):
        if num_queries is None: num_queries = self.num_queries
        
        if self.oracles.query_constrain().ranges is None:
            raise ValueError("Not for discrete Pools")
        else:
            a = self.oracles.query_constrain().queries_from_norm_pos(np.random.uniform(size=(num_queries, *self.oracles.query_constrain().shape)))
            return a

@dataclass
class LatinHypercubeQuerySampler(QuerySampler):

    def post_init(self):
        super().post_init()
        dim = 1
        for size in self.oracles.query_constrain().shape:
            dim *= size

        self.sampler = qmc.LatinHypercube(d=dim)


    def sample(self, num_queries = None):
        if num_queries is None: num_queries = self.num_queries

        if self.oracles.query_constrain().ranges is None:
            raise ValueError("Not for discrete Pools")
        else:
            sample = self.sampler.random(n=num_queries)
            
            sample = np.reshape(sample, (num_queries, *self.oracles.query_constrain().shape))

            a = self.oracles.query_constrain().queries_from_norm_pos(sample)
            return a

class RandomChoiceQuerySampler(QuerySampler):

    def sample(self, num_queries = None):
        if num_queries is None: num_queries = self.num_queries
        
        if self.oracles.query_constrain().count is None:
            raise ValueError("Not for continues pools")
        else:
            count = self.oracles.query_constrain().count
            if count == 0:
                return np.asarray([], dtype=np.int32)
            return self.oracles.query_constrain().queries_from_index(np.random.randint(low = 0, high = count, size=(num_queries,)))


class ProcessQuerySampler(QuerySampler):

    def post_init(self):
        super().post_init()
        if not isinstance(super().oracles, POracles):
            raise TypeError("ProcessQuerySampler requires POracles")

    @property
    def oracles(self) -> POracles:
        oracles: POracles = super().oracles
        return oracles

@dataclass
class LastProcessQuerySampler(ProcessQuerySampler):
    num_queries: int = init(default=None)

    def sample(self, num_queries = None):
        if num_queries is None: num_queries = self.num_queries

        return self.oracles.process.latest_add

@dataclass
class ProcessQueueQuerySampler(ProcessQuerySampler):
    num_queries: int = init(default=None)

    def sample(self, num_queries = None):
        if num_queries is None: num_queries = self.num_queries
        
        return self.oracles.process.queries
    

@dataclass
class DataPoolQuerySampler(QuerySampler):
    num_queries: int = init(default=None)

    def pool(self) -> QueriedDataPool:
        raise NotImplementedError("Please use a non abstract ...PoolQuerySampler.")

@dataclass
class AllDataPoolQuerySampler(DataPoolQuerySampler):
    num_queries: int = init(default=None)

    def sample(self, num_queries = None):
        if num_queries is None: num_queries = self.num_queries
        return self.pool().queries

@dataclass
class AllResultPoolQuerySampler(AllDataPoolQuerySampler):
    def post_init(self):
        super().post_init()
        if not isinstance(self.data_pools, ResultDataPools):
            raise TypeError("ResultPoolQuerySampler requires ResultDataPools")
    
    @property
    def data_pools(self) -> ResultDataPools:
        return super().data_pools

    def pool(self) -> QueriedDataPool:
        return self.data_pools.result

@dataclass
class AllStreamPoolQuerySampler(AllDataPoolQuerySampler):
    def post_init(self):
        super().post_init()
        if not isinstance(self.data_pools, StreamDataPools):
            raise TypeError("StreamPoolQuerySampler requires StreamDataPools")
    
    @property
    def data_pools(self) -> StreamDataPools:
        return super().data_pools

    def pool(self) -> QueriedDataPool:
        return self.data_pools.stream
    
@dataclass
class AllProcessPoolQuerySampler(AllDataPoolQuerySampler):
    def post_init(self):
        super().post_init()
        if not isinstance(self.data_pools, ProcessDataPools):
            raise TypeError("ProcessPoolQuerySampler requires ProcessDataPools")
    
    @property
    def data_pools(self) -> ProcessDataPools:
        return super().data_pools

    def pool(self) -> QueriedDataPool:
        return self.data_pools.process
    
@dataclass
class LastDataPoolQuerySampler(DataPoolQuerySampler):
    num_queries: int = init(default=None)

    def sample(self, num_queries = None):
        if num_queries is None: num_queries = self.num_queries
        return self.pool().last_queries

@dataclass
class LastResultPoolQuerySampler(LastDataPoolQuerySampler):
    def post_init(self):
        super().post_init()
        if not isinstance(self.data_pools, ResultDataPools):
            raise TypeError("ResultPoolQuerySampler requires ResultDataPools")
    
    @property
    def data_pools(self) -> ResultDataPools:
        return super().data_pools

    def pool(self) -> QueriedDataPool:
        return self.data_pools.result

@dataclass
class LastStreamPoolQuerySampler(LastDataPoolQuerySampler):
    def post_init(self):
        super().post_init()
        if not isinstance(self.data_pools, StreamDataPools):
            raise TypeError("StreamPoolQuerySampler requires StreamDataPools")
    
    @property
    def data_pools(self) -> StreamDataPools:
        return super().data_pools

    def pool(self) -> QueriedDataPool:
        return self.data_pools.stream
    
@dataclass
class LastProcessPoolQuerySampler(LastDataPoolQuerySampler):
    def post_init(self):
        super().post_init()
        if not isinstance(self.data_pools, ProcessDataPools):
            raise TypeError("ProcessPoolQuerySampler requires ProcessDataPools")
    
    @property
    def data_pools(self) -> ProcessDataPools:
        return super().data_pools

    def pool(self) -> QueriedDataPool:
        return self.data_pools.process