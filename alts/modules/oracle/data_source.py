from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic

from alts.core.data.data_pool import DataPool
from alts.core.oracle.data_source import DataSource
from alts.core.query.query_pool import QueryPool

if TYPE_CHECKING:
    from typing import Tuple, List, Any
    from alts.core.oracle.interpolation_strategy import InterpolationStrategy
    from alts.core.data_sampler import DataSampler

    from nptyping import  NDArray, Number, Shape

    from typing_extensions import Self




@dataclass
class LineDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    a: float = 1
    b: float = 0


    def query(self, queries):
        results = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*self.a) + np.ones(self.result_shape)*self.b
        return queries, results

    @property
    def query_pool(self) -> QueryPool:
        x_min = 0
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)

    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)


@dataclass
class SquareDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    x0: float = 0.5
    y0: float = 0
    s: float = 5

    def query(self, queries):
        results = np.dot((queries - self.x0)**2, np.ones((*self.query_shape,*self.result_shape))*self.s) + np.ones(self.result_shape)*self.y0
        return queries, results

    @property
    def query_pool(self) -> QueryPool:
        x_min = 0
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)
    
    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)

    
@dataclass
class InterpolatingDataSource(DataSource):
    data_sampler: DataSampler
    interpolation_strategy: InterpolationStrategy

    def query(self, queries):
        data_points = self.data_sampler.sample(queries)
        data_points = self.interpolation_strategy.interpolate(data_points)
        return data_points

    @property
    def query_pool(self) -> QueryPool:
        return self.interpolation_strategy.query_pool

    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)


@dataclass
class CrossDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    a: float = 1

    def query(self, queries):

        direction = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results_up = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*self.a) 
        results_down = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*(-self.a))

        results = (1- direction)*results_up + direction*results_down

        return queries, results

    @property
    def query_pool(self) -> QueryPool:
        x_min = -0.5
        x_max = 0.5
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)
    
    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)

@dataclass
class DoubleLinearDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    a: float = 1
    slope_factor = 0.5

    def query(self, queries):

        slope = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results_steap = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*self.a) 
        results_flat = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*self.slope_factor*self.a)

        results = (1- slope)*results_steap + slope*results_flat

        return queries, results

    @property
    def query_pool(self) -> QueryPool:
        x_min = -0.5
        x_max = 0.5
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)
    
    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)

        
@dataclass
class HourglassDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    a: float = 1

    def query(self, queries):

        kind = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results_up = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*self.a) 
        results_down = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*(-self.a))

        results_dir = (1- kind)*results_up + kind*results_down
        results_const = (1- kind)*0.5 + kind*-0.5

        const = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results = (1- const)*results_dir + const*results_const

        return queries, results

    @property
    def query_pool(self) -> QueryPool:
        x_min = -0.5
        x_max = 0.5
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)
    
    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)

@dataclass
class ZDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    a: float = 1

    def query(self, queries):

        kind = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results_const = (1- kind)*0.5 + kind*-0.5
        results_up = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*self.a) 


        const = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results = (1- const)*results_up + const*results_const

        return queries, results

    @property
    def query_pool(self) -> QueryPool:
        x_min = -0.5
        x_max = 0.5
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)
    
    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)

@dataclass
class ZInvDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    a: float = 1

    def query(self, queries):

        kind = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results_const = (1- kind)*0.5 + kind*-0.5
        results_down = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*(-self.a))

        const = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results = (1- const)*results_down + const*results_const

        return queries, results

    @property
    def query_pool(self) -> QueryPool:
        x_min = -0.5
        x_max = 0.5
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)
    
    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)

@dataclass
class LinearPeriodicDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    a: float = 1
    p: float = 0.2


    def query(self, queries):
        results = np.dot(queries % self.p, np.ones((*self.query_shape, *self.result_shape))*self.a)
        return queries, results

    @property
    def query_pool(self) -> QueryPool:
        x_min = 0
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)

    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)

@dataclass
class SineDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    a: float = 1
    p: float = 1
    y0: float = 0
    x0: float = 0


    def query(self, queries):
        results = np.dot(np.sin((queries-self.x0) * 2 * np.pi * self.p), np.ones((*self.query_shape,*self.result_shape))*self.a) + np.ones(self.result_shape)*self.y0
        return queries, results

    @property
    def query_pool(self) -> QueryPool:
        x_min = 0
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)

    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)


@dataclass
class HypercubeDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    w: float = 0.3

    def query(self, queries):

        kind = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        random = np.random.uniform(-0.5,0.5, size=(queries.shape[0], *self.result_shape))

        results_const = (1- kind)*0.5 + kind*-0.5

        mask = np.greater(queries, -self.w) * np.less(queries, self.w)

        results = mask * results_const + (1 - mask) * random

        return queries, results


    @property
    def query_pool(self) -> QueryPool:
        x_min = -0.5
        x_max = 0.5
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)
    
    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)

@dataclass
class StarDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    w: float = 0.05

    def query(self, queries):

        direction = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results_up = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))) 
        results_down = np.dot(queries, -np.ones((*self.query_shape,*self.result_shape)))

        results_dir = (1- direction)*results_up + direction*results_down

        const = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        result_const = const*results_dir

        random = np.random.uniform(-0.5,0.5, size=(queries.shape[0], *self.result_shape))

        mask = np.greater(queries, -self.w) * np.less(queries, self.w)

        results = mask * random + (1 - mask) * result_const

        return queries, results

    @property
    def query_pool(self) -> QueryPool:
        x_min = -0.5
        x_max = 0.5
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)
    
    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)


from threading import Lock
@dataclass
class GausianProcessDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    s: float = 0.1
    r: int = np.random.randint(1, 10000000)

    gpr: GaussianProcessRegressor = field(default=None,init=False)
    queries: NDArray[Number, Shape["query_nr, ... query_dim"]] = field(default=None,init=False)
    results: NDArray[Number, Shape["query_nr, ... result_dim"]] = field(default=None,init=False)
    singleton: GaussianProcessRegressor = field(default=None,init=False)

    def query(self, queries):
        results = self.gpr.sample_y(queries)
        self.queries = np.concatenate((self.queries, queries))
        self.results = np.concatenate((self.results, results))
        self.gpr.fit(self.queries, self.results)

        return queries, results

    @property
    def query_pool(self) -> QueryPool:
        x_min = 0
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)

    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)

    def __call__(self, **kwargs) -> Self:
        obj = super().__call__( **kwargs)
        if self.singleton is None:
            obj.gpr = GaussianProcessRegressor(kernel=RationalQuadratic(length_scale=obj.s), optimizer=None, random_state=obj.r)
            obj.queries = np.empty((0,*self.query_shape))
            obj.results = np.empty((0,*self.result_shape))
            obj.singleton = obj
            self.singleton = obj
        return self.singleton
    
