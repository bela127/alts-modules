#Version 1.1.1 conform as of 23.04.2025
"""
| *alts.modules.query.query_sampler*
| :doc:`Core Module </core/query/query_sampler>`
"""
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
    """
    OptimalQuerySampler(num_queries)
    | **Description**
    |   Samples randomly from a given list of "optimal" queries.

    :param num_queries: Number of queries to sample by default
    :type num_queries: int
    :param optimal_queries: What queries to sample from
    :type optimal_queries: Tuple[NDArray[Shape["query_nr, ... query_dims"], Number], ...]
    """
    optimal_queries: Tuple[NDArray[Shape["query_nr, ... query_dims"], Number], ...] = init() # type: ignore

    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Checks if all queries optimal_queries meet its oracle's query constraints.

        :raises ValueError: If any query in optimal_queries does not meet the query constraints.
        """
        super().post_init()
        for optimal_query in self.optimal_queries:
            if not self.oracles.query_constrain().constrains_met(optimal_query):
                raise ValueError("optimal_queries do not meet oracles.query_constrain")

    def sample(self, num_queries = None):
        """
        sample(self, num_queries) -> queries
        | **Description**
        |   Returns approximately ```num_queries``` queries randomly chosen among the ```optimal_queries```

        :param num_queries: Number of queries to sample (default= self.num_queries)
        :type num_queries: int
        :return: Sampled queries
        :rtype: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        """
        if num_queries is None: num_queries = self.num_queries
        
        query_nr = self.optimal_queries[0].shape[0]
        k = ceil(num_queries / query_nr) 
        queries = random.choices(self.optimal_queries, k=k)
        queries = np.concatenate(queries)
        return queries[:num_queries]

@dataclass
class FixedQuerySampler(QuerySampler):
    """
    FixedQuerySampler(num_queries)
    | **Description**
    |   Always samples the same fixed query.

    :param num_queries: Default number of queries to sample
    :type num_queries: int
    :param fixed_query: The fixed query that is always sampled
    :type fixed_query: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
    """
    fixed_query: NDArray[Shape["... query_dims"], Number] = init() # type: ignore

    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Checks if all queries optimal_queries meet its oracle's query constraints.

        :raises ValueError: If any query in optimal_queries does not meet the query constraints.
        """
        super().post_init()
        if not self.oracles.query_constrain().constrains_met(self.fixed_query):
            raise ValueError("fixed_query does not meet oracles.query_constrain")

    def sample(self, num_queries = None):
        """
        sample(self, num_queries) -> queries
        | **Description**
        |   Returns num_queries copies of fixed_query

        :param num_queries: Number of queries to sample (default= self.num_query)
        :type num_queries: int
        :return: Sampled queries 
        :rtype: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        """
        if num_queries is None: num_queries = self.num_queries
        
        queries = np.repeat(self.fixed_query[None, ...], num_queries, axis=0)
        return queries
    
@dataclass
class UniformQuerySampler(QuerySampler):
    """
    UniformQuerySampler(num_queries)
    | **Description**
    |   Samples queries with random values.

    :param num_queries: Number of queries to sample by default
    :type num_queries: int
    """
    def sample(self, num_queries = None):
        """
        sample(self, num_queries) -> queries
        | **Description**
        |   Returns queries with uniformly random values within the query constraints.

        :param num_queries: Number of queries to sample (default= self.num_queries)
        :type num_queries: int
        :return: Sampled queries
        :rtype: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :raises ValueError: If queries are constraint to discrete values
        """
        if num_queries is None: num_queries = self.num_queries
        
        if self.oracles.query_constrain().ranges is None:
            raise ValueError("Not for discrete Pools")
        else:
            a = self.oracles.query_constrain().queries_from_norm_pos(np.random.uniform(size=(num_queries, *self.oracles.query_constrain().shape)))
            return a

@dataclass
class LatinHypercubeQuerySampler(QuerySampler):
    """
    LatinHypercubeQuerySampler(num_queries)
    | **Description**
    |   Does Latin Hypercube sampling of queries.

    :param num_queries: Number of queries to sample by default
    :type num_queries: int
    """
    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Initializes the appropiate dimesnional quasi monte-carlo Latin Hypercube Sampler.
        """
        super().post_init()
        dim = 1
        for size in self.oracles.query_constrain().shape:
            dim *= size

        self.sampler = qmc.LatinHypercube(d=dim)


    def sample(self, num_queries = None):
        """
        sample(self, num_queries) -> queries
        | **Description**
        |   Returns queries with latin-hypercube random values within the query constraints.

        :param num_queries: Number of queries to sample (default= self.num_queries)
        :type num_queries: int
        :return: Sampled queries
        :rtype: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :raises ValueError: If queries are constraint to discrete values
        """
        if num_queries is None: num_queries = self.num_queries

        if self.oracles.query_constrain().ranges is None:
            raise ValueError("Not for discrete Pools")
        else:
            sample = self.sampler.random(n=num_queries)
            
            sample = np.reshape(sample, (num_queries, *self.oracles.query_constrain().shape))

            a = self.oracles.query_constrain().queries_from_norm_pos(sample)
            return a

class RandomChoiceQuerySampler(QuerySampler):
    """
    RandomChoiceQuerySampler(num_queries)
    | **Description**
    |   Samples queries from discrete constraint pools.

    :param num_queries: Number of queries to sample by default
    :type num_queries: int
    """
    def sample(self, num_queries = None):
        """
        sample(self, num_queries) -> queries
        | **Description**
        |   Returns randomly chosen queries from the constraint-permitted pool of queires.

        :param num_queries: Number of queries to sample (default= self.num_queries)
        :type num_queries: int
        :return: Sampled queries
        :rtype: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :raises ValueError: If queries are constraint to continuous values
        """
        if num_queries is None: num_queries = self.num_queries
        
        if self.oracles.query_constrain().count is None:
            raise ValueError("Not for continues pools")
        else:
            count = self.oracles.query_constrain().count
            if count == 0:
                return np.asarray([], dtype=np.int32)
            return self.oracles.query_constrain().queries_from_index(np.random.randint(low = 0, high = count, size=(num_queries,)))

class ProcessQuerySampler(QuerySampler):
    """
    ProcessQuerySampler(num_queries)
    | **Description**
    |   Samples queries from a Process Oracle in some way.
    |   This class is abstract.

    :param num_queries: Number of queries to sample by default
    :type num_queries: int
    """
    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Raises TypeError if Oracles is not PORacles

        :raises TypeError: If Oracles is not PORacles
        """
        super().post_init()
        if not isinstance(super().oracles, POracles):
            raise TypeError("ProcessQuerySampler requires POracles")

    @property
    def oracles(self) -> POracles:
        """
        oracles(self) -> POracles
        | **Description**
        |   Returns the sampler's oracle.

        :return: QuerySampler's oracle
        :rtype: POracles
        """
        oracles: POracles = super().oracles # type: ignore
        return oracles

@dataclass
class LastProcessQuerySampler(ProcessQuerySampler):
    """
    LastProcessQuerySampler(num_queries)
    | **Description**
    |   Samples the queries last added to the Process Oracle.

    :param num_queries: Number of queries to sample by default (ignored in this class)
    :type num_queries: int
    """
    num_queries: int = init(default=None)

    def sample(self, num_queries = None):
        """
        sample(self, num_queries) -> queries
        | **Description**
        |   Returns the queries last added to the Procass Oracle.

        :param num_queries: Unused
        :type num_queries: Any
        :return: Sampled queries
        :rtype: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        """
        if num_queries is None: num_queries = self.num_queries

        return self.oracles.process.latest_add

@dataclass
class ProcessQueueQuerySampler(ProcessQuerySampler):
    """
    ProcessQueueQuerySampler(num_queries)
    | **Description**
    |   Samples *all* queries in the Process Oracle's query queue.

    :param num_queries: Number of queries to sample by default (ignored in this class)
    :type num_queries: int
    """
    num_queries: int = init(default=None)

    def sample(self, num_queries = None):
        """
        sample(self, num_queries) -> queries
        | **Description**
        |   Returns all queries in the query queue of the Procass Oracle.

        :param num_queries: Unused
        :type num_queries: Any
        :return: Sampled queries
        :rtype: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        """
        if num_queries is None: num_queries = self.num_queries
        
        return self.oracles.process.queries
    

@dataclass
class DataPoolQuerySampler(QuerySampler):
    """
    DataPoolQuerySampler(num_queries)
    | **Description**
    |   Samples from Queried Data Pools in some way (this class is abstract).

    :param num_queries: Number of queries to sample by default (ignored in this class)
    :type num_queries: int
    """
    num_queries: int = init(default=None)

    def pool(self) -> QueriedDataPool:
        """
        pool(self) -> QueriedDataPool
        | **Description**
        |   Returns the sampler's data pool (abstract).

        :return: This Query Sampler's data pool
        :rtype: QueriedDataPool
        :raises NotImplementedError: This class is abstract
        """
        raise NotImplementedError("Please use a non abstract ...PoolQuerySampler.")

@dataclass
class AllDataPoolQuerySampler(DataPoolQuerySampler):
    #TODO Should pool() be abstract here?
    """
    AllDataPoolQuerySampler(num_queries)
    | **Description**
    |   Samples all of the Data Pool's data (asbtract).

    :param num_queries: Number of queries to sample by default (ignored in this class)
    :type num_queries: int
    """
    num_queries: int = init(default=None)

    def sample(self, num_queries = None):
        """
        sample(self, num_queries) -> queries
        | **Description**
        |   Returns all queries in the data pool.

        :param num_queries: Unused
        :type num_queries: Any
        :return: Sampled queries
        :rtype: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        """
        if num_queries is None: num_queries = self.num_queries
        return self.pool().queries

@dataclass
class AllResultPoolQuerySampler(AllDataPoolQuerySampler):
    """
    AllResultPoolQuerySampler(num_queries)
    | **Description**
    |   Samples the entire ResultDataPools.

    :param num_queries: Number of queries to sample by default (ignored in this class)
    :type num_queries: int
    """
    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Raises TypeError if DataPools is not ResultDataPools

        :raises TypeError: If DataPools is not ResultDataPools
        """
        super().post_init()
        if not isinstance(self.data_pools, ResultDataPools):
            raise TypeError("ResultPoolQuerySampler requires ResultDataPools")
    
    @property
    def data_pools(self) -> ResultDataPools:
        """
        data_pools(self) -> ResultDataPools
        | **Description**
        |   Returns the sampler's DataPools.

        :return: QuerySampler's DataPools
        :rtype: ResultDataPools
        """
        return super().data_pools # type: ignore

    def pool(self) -> QueriedDataPool:
        """
        pool(self) -> QueriedDataPool
        | **Description**
        |   Returns the sampler's data pools as a queryable.

        :return: Sampler's queryable data pool
        :rtype: QueriedDataPool
        """
        return self.data_pools.result

@dataclass
class AllStreamPoolQuerySampler(AllDataPoolQuerySampler):
    """
    AllStreamPoolQuerySampler(num_queries)
    | **Description**
    |   Samples the entire StreamDataPools.

    :param num_queries: Number of queries to sample by default (ignored in this class)
    :type num_queries: int
    """
    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Raises TypeError if DataPools is not StreamDataPools

        :raises TypeError: If DataPools is not StreamDataPools
        """
        super().post_init()
        if not isinstance(self.data_pools, StreamDataPools):
            raise TypeError("StreamPoolQuerySampler requires StreamDataPools")
    
    @property
    def data_pools(self) -> StreamDataPools:
        """
        data_pools(self) -> StreamDataPools
        | **Description**
        |   Returns the sampler's DataPools.

        :return: QuerySampler's DataPools
        :rtype: StreamDataPools
        """
        return super().data_pools # type: ignore

    def pool(self) -> QueriedDataPool:
        """
        pool(self) -> QueriedDataPool
        | **Description**
        |   Returns the sampler's data pools as a queryable.

        :return: Sampler's queryable data pool
        :rtype: QueriedDataPool
        """
        return self.data_pools.stream
    
@dataclass
class AllProcessPoolQuerySampler(AllDataPoolQuerySampler):
    """
    AllProcessPoolQuerySampler(num_queries)
    | **Description**
    |   Samples the entire ProcessDataPools.

    :param num_queries: Number of queries to sample by default (ignored in this class)
    :type num_queries: int
    """
    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Raises TypeError if DataPools is not ProcessDataPools

        :raises TypeError: If DataPools is not ProcessDataPools
        """
        super().post_init()
        if not isinstance(self.data_pools, ProcessDataPools):
            raise TypeError("ProcessPoolQuerySampler requires ProcessDataPools")
    
    @property
    def data_pools(self) -> ProcessDataPools:
        """
        data_pools(self) -> ProcessDataPools
        | **Description**
        |   Returns the sampler's DataPools.

        :return: QuerySampler's DataPools
        :rtype: ProcessDataPools
        """
        return super().data_pools # type: ignore

    def pool(self) -> QueriedDataPool:
        """
        pool(self) -> QueriedDataPool
        | **Description**
        |   Returns the sampler's data pools as a queryable.

        :return: Sampler's queryable data pool
        :rtype: QueriedDataPool
        """
        return self.data_pools.process
    
@dataclass
class LastDataPoolQuerySampler(DataPoolQuerySampler):
    """
    LastDataPoolQuerySampler(num_queries)
    | **Description**
    |   Samples the DataPool's last added queries

    :param num_queries: Number of queries to sample by default (ignored in this class)
    :type num_queries: int
    """
    num_queries: int = init(default=None)

    def sample(self, num_queries = None):
        """
        sample(self, num_queries) -> queries
        | **Description**
        |   Returns the DataPool's last added queries.

        :param num_queries: Unused
        :type num_queries: Any
        :return: Sampled queries
        :rtype: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        """
        if num_queries is None: num_queries = self.num_queries
        return self.pool().last_queries

@dataclass
class LastResultPoolQuerySampler(LastDataPoolQuerySampler):
    """
    LastResultPoolQuerySampler(num_queries)
    | **Description**
    |   Samples the ResultDataPool's last added queries

    :param num_queries: Number of queries to sample by default (ignored in this class)
    :type num_queries: int
    """
    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Raises TypeError if DataPools is not ResultDataPools

        :raises TypeError: If DataPools is not ResultDataPools
        """
        super().post_init()
        if not isinstance(self.data_pools, ResultDataPools):
            raise TypeError("ResultPoolQuerySampler requires ResultDataPools")
    
    @property
    def data_pools(self) -> ResultDataPools:
        """
        data_pools(self) -> ResultDataPools
        | **Description**
        |   Returns the sampler's DataPools.

        :return: QuerySampler's DataPools
        :rtype: ResultDataPools
        """
        return super().data_pools # type: ignore

    def pool(self) -> QueriedDataPool:
        """
        pool(self) -> QueriedDataPool
        | **Description**
        |   Returns the sampler's data pools as a queryable.

        :return: Sampler's queryable data pool
        :rtype: QueriedDataPool
        """
        return self.data_pools.result

@dataclass
class LastStreamPoolQuerySampler(LastDataPoolQuerySampler):
    """
    LastStreamPoolQuerySampler(num_queries)
    | **Description**
    |   Samples the StreamDataPool's last added queries

    :param num_queries: Number of queries to sample by default (ignored in this class)
    :type num_queries: int
    """
    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Raises TypeError if DataPools is not StreamDataPools

        :raises TypeError: If DataPools is not StreamDataPools
        """
        super().post_init()
        if not isinstance(self.data_pools, StreamDataPools):
            raise TypeError("StreamPoolQuerySampler requires StreamDataPools")
    
    @property
    def data_pools(self) -> StreamDataPools:
        """
        data_pools(self) -> StreamDataPools
        | **Description**
        |   Returns the sampler's DataPools.

        :return: QuerySampler's DataPools
        :rtype: StreamDataPools
        """
        return super().data_pools # type: ignore

    def pool(self) -> QueriedDataPool:
        """
        pool(self) -> QueriedDataPool
        | **Description**
        |   Returns the sampler's data pools as a queryable.

        :return: Sampler's queryable data pool
        :rtype: QueriedDataPool
        """
        return self.data_pools.stream
    
@dataclass
class LastProcessPoolQuerySampler(LastDataPoolQuerySampler):
    """
    LastStreamPoolQuerySampler(num_queries)
    | **Description**
    |   Samples the ProcessDataPool's last added queries

    :param num_queries: Number of queries to sample by default (ignored in this class)
    :type num_queries: int
    """
    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Raises TypeError if DataPools is not ProcessDataPools

        :raises TypeError: If DataPools is not ProcessDataPools
        """
        super().post_init()
        if not isinstance(self.data_pools, ProcessDataPools):
            raise TypeError("ProcessPoolQuerySampler requires ProcessDataPools")
    
    @property
    def data_pools(self) -> ProcessDataPools:
        """
        data_pools(self) -> ProcessDataPools
        | **Description**
        |   Returns the sampler's DataPools.

        :return: QuerySampler's DataPools
        :rtype: ProcessDataPools
        """
        return super().data_pools # type: ignore

    def pool(self) -> QueriedDataPool:
        """
        pool(self) -> QueriedDataPool
        | **Description**
        |   Returns the sampler's data pools as a queryable.

        :return: Sampler's queryable data pool
        :rtype: QueriedDataPool
        """
        return self.data_pools.process