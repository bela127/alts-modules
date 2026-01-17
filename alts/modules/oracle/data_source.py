#Version 1.1.1 conform as of 06.05.2025
"""
| *alts.modules.oracle.data_source*
| :doc:`Core Module </core/oracle/data_source>`
"""
from __future__ import annotations
from math import floor
from typing import TYPE_CHECKING, Optional

from dataclasses import dataclass, field, InitVar

import numpy as np
import GPy

from alts.core.oracle.data_source import DataSource, TimeDataSource
from alts.core.data.constrains import QueryConstrain, ResultConstrain

from alts.core.configuration import pre_init, is_set, init, post_init

if TYPE_CHECKING:
    from alts.core.oracle.data_behavior import DataBehavior
    from alts.core.configuration import Required

    from typing import Tuple, List, Any, Type
    from alts.core.oracle.interpolation_strategy import InterpolationStrategy
    from alts.core.data.data_sampler import DataSampler

    from nptyping import  NDArray, Number, Shape

    from typing_extensions import Self


@dataclass
class RandomUniformDataSource(DataSource):
    """
    RandomUniformDataSource(query_shape, result_shape, u, l)
    | **Description**
    |   A ``RandomUniformDataSource`` is an **independent** source of data.
    |   For more details see `numpy.random.uniorm <https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html>`_.

    :param query_shape: The expected shape of the queries (default= (1,))
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results (default= (1,))
    :type result_shape: tuple of ints
    :param u: The upper bound of query values (exclusive) (default= 1)
    :type u: float
    :param l: The lower bound of query values (inclusive), (default= 0)
    :type l: float
    """
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    u: float = init(default=1)
    l: float = init(default=0)

    def query(self, queries):
        """
        query(self, queries) -> data_points
        | **Description**
        |   Each query receives a uniformly random result in [l, u).

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  
        """
        results = np.random.uniform(low=self.l, high=self.u, size=(queries.shape[0], * self.result_shape))
        return queries, results

    def query_constrain(self) -> QueryConstrain:
        """
        query_constrain(self) -> QueryConstrain
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [0, 1)

        :return: Constrains around queries
        :rtype: QueryConstrain
        """
        x_min = 0
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(np.prod(self.query_shape)))).reshape((*self.query_shape,2))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)
    
    def result_constrain(self) -> ResultConstrain:
        """
        result_constrain(self) -> ResultConstrain
        | **Description**
        |   See :func:`DataSource.result_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``result_shape``
        |   *Value Range:* [l, u)

        :return: Constrains around results
        :rtype: ResultConstrain
        """
        y_min = 0
        y_max = 1
        result_ranges = np.asarray(tuple((y_min, y_max) for i in range(self.result_shape[0])))
        return ResultConstrain(shape=self.result_shape, ranges=result_ranges)

@dataclass
class LineDataSource(DataSource):
    """
    LineDataSource(query_shape, result_shape, a, b)
    | **Description**
    |   A ``LineDataSource`` is a **deterministic** source of data representing a linear equation ``y = ax + b``.

    :param query_shape: The expected shape of the queries (default= (1,))
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results (default= (1,))
    :type result_shape: tuple of ints
    :param a: Coefficient of degree 1, (default= 1)
    :type a: float (optional)
    :param b: Coefficient of degree 0, (default= 0)
    :type b: float (optional)
    """
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    a: float = init(default=1)
    b: float = init(default=0)

    def query(self, queries):
        """
        query(self, queries) -> data_points
        | **Description**
        |   See :func:`DataSource.query()`

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  
        """
         
        results = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*self.a) + np.ones(self.result_shape)*self.b
        return queries, results

    def query_constrain(self) -> QueryConstrain:
        """
        query_constrain(self) -> QueryConstrain
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [0, 1)

        :return: Constrains around queries
        :rtype: QueryConstrain
        """
        x_min = 0
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)
    
    def result_constrain(self) -> ResultConstrain:
        """
        result_constrain(self) -> ResultConstrain
        | **Description**
        |   See :func:`DataSource.result_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``result_shape``
        |   *Value Range:*

        +-----+-------+--------+-----+--------+-------+
        | MIN | a < 0 | a >= 0 | MAX | a <= 0 | a > 0 |
        +=====+=======+========+=====+========+=======+
        |     | a + b | b      |     | b      | a + b |
        +-----+-------+--------+-----+--------+-------+

        :return: Constrains around results
        :rtype: ResultConstrain
        """
        y_min = self.a + self.b if self.a < 0 else self.b
        y_max = self.b if self.a <= 0 else self.a + self.b
        result_ranges = np.asarray(tuple((y_min, y_max) for i in range(self.result_shape[0])))
        return ResultConstrain(shape=self.result_shape, ranges=result_ranges)

@dataclass
class SquareDataSource(DataSource):
    """
    SquareDataSource(query_shape, result_shape, x0, y0, s)
    | **Description**
    |   A ``SquareDataSource`` is a **deterministic** source of data representing a square parabola ``s * (x - x0)^2 + y0``. 

    :param query_shape: The expected shape of the queries (default= (1,))
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results (default= (1,))
    :type result_shape: tuple of ints
    :param x0: Offset of the parabola in x-direction, (default= 0.5)
    :type x0: float (optional)
    :param y0: Offset of the parabola in y-direction, (default= 0)
    :type y0: float (optional)
    :param s: Coefficient of degree 2, (default= 5)
    :type s: float (optional)
    """
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    x0: float = init(default=0.5)
    y0: float = init(default=0)
    s: float = init(default=5)

    def query(self, queries):
        """
        query(self, queries) -> data_points
        | **Description**
        |   See :func:`DataSource.query()`

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  
        """
         
        results = np.dot((queries - self.x0)**2, np.ones((*self.query_shape,*self.result_shape))*self.s) + np.ones(self.result_shape)*self.y0
        return queries, results

    def query_constrain(self) -> QueryConstrain:
        """
        query_constrain(self) -> QueryConstrain
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [0, 1)

        :return: Constrains around queries
        :rtype: QueryConstrain
        """
        x_min = 0
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)
    
    def result_constrain(self) -> ResultConstrain:
        """
        result_constrain(self) -> ResultConstrain
        | **Description**
        |   See :func:`DataSource.result_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``result_shape``
        |   *Value Range:*

        +-------------+---------+------+-------------+-------------+-------------+
        |MIN          |s < 0    |s >= 0|MAX          |s < 0        |s >= 0       |
        +=============+=========+======+=============+=============+=============+
        |x0 < 0       |s*x0^2+y0|y0    |x0 < 0       |s*x0^2+y0    |s*(1-x0)^2+y0|
        +-------------+---------+------+-------------+-------------+-------------+
        |0 <= x0 < 0.5|N/A      |N/A   |0 <= x0 < 0.5|y0           |s*(1-x0)^2+y0|
        +-------------+---------+------+-------------+-------------+-------------+
        |0.5 <= x0 < 1|N/A      |N/A   |0.5 <= x0 < 1|y0           |s*x0^2+y0    |
        +-------------+---------+------+-------------+-------------+-------------+
        |1 <= x0      |N/A      |N/A   |1 <= x0      |s*(1-x0)^2+y0|s*x0^2+y0    |
        +-------------+---------+------+-------------+-------------+-------------+

        :return: Constrains around results
        :rtype: ResultConstrain
        """
        y_min = self.s*self.x0**2+self.y0 if self.s<0 else self.y0
        y_max = self.y0 if (self.s<0 and 0<=self.x0 and self.x0<1) else self.s*self.x0**2+self.y0 if (self.x0<0 and self.s<0 or self.s>=0 and self.x0>=0.5) else self.s*(1-self.x0)**2+self.y0
        result_ranges = np.asarray(tuple((y_min, y_max) for i in range(self.result_shape[0])))
        return ResultConstrain(shape=self.result_shape, ranges=result_ranges)

@dataclass
class PowDataSource(DataSource):
    """
    PowDataSource(query_shape, result_shape, p, s)
    | **Description**
    |   A ``PowDataSource`` is a **deterministic** source of data representing an exponential equation ``s * x^p``. 

    :param query_shape: The expected shape of the queries (default= (1,))
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results (default= (1,))
    :type result_shape: tuple of ints
    :param p: Power of x (default= 3)
    :type p: float
    :param s: Coefficient of x^power (default= 1)
    :type s: float
    """
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    p: float = init(default=3)
    s: float = init(default=1)

    def query(self, queries):
        """
        query(self, queries) -> data_points
        | **Description**
        |   See :func:`DataSource.query()`

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  
        """
         
        results = np.dot(np.power(queries, self.p), np.ones((*self.query_shape,*self.result_shape))*self.s)
        return queries, results

    def query_constrain(self) -> QueryConstrain:
        """
        query_constrain(self) -> QueryConstrain
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [0, 1)

        :return: Constrains around queries
        :rtype: QueryConstrain
        """
        x_min = 0
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)
    
    def result_constrain(self) -> ResultConstrain:
        """
        result_constrain(self) -> ResultConstrain
        | **Description**
        |   See :func:`DataSource.result_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``result_shape``
        |   *Value Range:*

        +------+-----+-----+-----+------+-----+-----+-----+
        |MIN   |p < 0|p = 0|p > 0|MAX   |p < 0|p = 0|p > 0|  
        +======+=====+=====+=====+======+=====+=====+=====+
        |s < 0 |-inf |s    |s    |s < 0 |s    |s    |0    |  
        +------+-----+-----+-----+------+-----+-----+-----+
        |s >= 0|s    |s    |0    |s >= 0|inf  |s    |s    |  
        +------+-----+-----+-----+------+-----+-----+-----+

        :return: Constrains around results
        :rtype: ResultConstrain
        """
        y_min = self.s if (self.s<0 and self.p>=0 or self.s>=0 and self.p<=0) else 0 if (self.p>0 and self.s>=0) else np.NINF
        y_max = self.s if (self.s<0 and self.p<=0 or self.s>=0 and self.p>=0) else 0 if (self.p>0 and self.s<0) else np.INF # type: ignore
        result_ranges = np.asarray(tuple((y_min, y_max) for i in range(self.result_shape[0])))
        return ResultConstrain(shape=self.result_shape, ranges=result_ranges)

@dataclass
class ExpDataSource(DataSource):
    """
    ExpDataSource(query_shape, result_shape, b, s)
    | **Description**
    |   An ``ExpDataSource`` is a **deterministic** source of data representing an exponential equation ``s * b^x``. 

    :param query_shape: The expected shape of the queries (default= (1,))
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results (default= (1,))
    :type result_shape: tuple of ints
    :param b: Basis to the exponent x (default= 2)
    :type b: float
    :param s: Coefficient of base^x (default= 1)
    :type s: float
    """
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    b: float = init(default=2)
    s: float = init(default=1)

    def query(self, queries):
        """
        query(self, queries) -> data_points
        | **Description**
        |   See :func:`DataSource.query()`

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  
        """
         
        results = np.dot(np.power(self.b, queries*self.s), np.ones((*self.query_shape,*self.result_shape)))
        return queries, results

    def query_constrain(self) -> QueryConstrain:
        """
        query_constrain(self) -> QueryConstrain
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [0, 1)

        :return: Constrains around queries
        :rtype: QueryConstrain
        """
        x_min = 0
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)
    
    def result_constrain(self) -> ResultConstrain:
        """
        result_constrain(self) -> ResultConstrain
        | **Description**
        |   See :func:`DataSource.result_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``result_shape``
        |   *Value Range:* 

        +------+-----+-----+----------+-----+-----+------+-----+-----+----------+-----+-----+
        |MIN   |b < 0|b = 0|0 <= b < 1|b = 1|b > 1|MAX   |b < 0|b = 0|0 <= b < 1|b = 1|b > 1|
        +======+=====+=====+==========+=====+=====+======+=====+=====+==========+=====+=====+
        |s < 0 |NaN  |s    |s         |s    |s * b|s < 0 |NaN  |s    |s * b     |s    |s    |
        +------+-----+-----+----------+-----+-----+------+-----+-----+----------+-----+-----+
        |s = 0 |NaN  |0    |0         |0    |0    |s = 0 |NaN  |0    |0         |0    |0    |
        +------+-----+-----+----------+-----+-----+------+-----+-----+----------+-----+-----+
        |s >= 0|NaN  |0    |s * b     |s    |s    |s >= 0|NaN  |s    |s         |s    |s * b|
        +------+-----+-----+----------+-----+-----+------+-----+-----+----------+-----+-----+

        :return: Constrains around results
        :rtype: ResultConstrain
        """
        y_min = self.s if (self.s<0 and self.b>=0 and self.b<1 or self.s>0 and self.b>=1) else self.s*self.b if (self.s==0 or self.s>=0 and self.b<1 or self.s<0 and self.b>=1) else -self.s*self.b
        y_max = self.s if (self.s>=0 and self.b>=1 and self.b<1 or self.s<0 and self.b>=1) else self.s*self.b if (self.s==0 or self.s<0 and self.b<1 or self.s>=0 and self.b>=1) else -self.s*self.b
        result_ranges = np.asarray(tuple((y_min, y_max) for i in range(self.result_shape[0])))
        return ResultConstrain(shape=self.result_shape, ranges=result_ranges)

@dataclass
class InterpolatingDataSource(DataSource):
    """
    InterpolatingDataSource(data_sampler, interpolation_strategy)
    | **Description**
    |   An ``InterpolatingDataSource`` is an **independent** source of data depending on the :doc:`DataSource </core/oracle/data_source>` it interpolates within. 

    :param data_sampler: The data sampler to sample data for interpolation
    :type data_sampler: DataSampler
    :param interpolation_strategy: The interpolation strategy for the data points
    :type interpolation_strategy: InterpolationStrategy
    """
    data_sampler: DataSampler = init()
    interpolation_strategy: InterpolationStrategy = init()

    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Initializes its DataSampler and InterpolationStrategy.
        """
        super().post_init()
        self.data_sampler = self.data_sampler()
        self.interpolation_strategy = self.interpolation_strategy(self.data_sampler)

    def query(self, queries):
        """
        query(self, queries) -> data_points
        | **Description**
        |   See :func:`DataSource.query()`

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  
        """
        data_points = self.data_sampler.query(queries)
        data_points = self.interpolation_strategy.interpolate(data_points)
        return data_points
    
    def query_constrain(self) -> QueryConstrain:
        """
        query_constrain(self) -> QueryConstrain
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        :return: Constrains around queries
        :rtype: QueryConstrain
        """
        return self.interpolation_strategy.query_constrain()

@dataclass
class CrossDataSource(DataSource):
    """
    CrossDataSource(query_shape, result_shape, a)
    | **Description**
    |   A ``CrossDataSource`` is a **function-prior random** source of data choosing one of the following equations at random {``-a * x``, ``a * x``}. 

    :param query_shape: The expected shape of the queries (default= (1,))
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results (default= (1,))
    :type result_shape: tuple of ints
    :param a: Coefficient of x (default= 1)
    :type a: float
    """
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    a: float = init(default=1)

    def query(self, queries):
        """
        query(self, queries) -> data_points
        | **Description**
        |   See :func:`DataSource.query()`

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  
        """
         
        direction = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results_up = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*self.a) 
        results_down = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*(-self.a))

        results = (1- direction)*results_up + direction*results_down

        return queries, results

    def query_constrain(self) -> QueryConstrain:
        """
        query_constrain(self) -> QueryConstrain
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [-0.5, 0.5)

        :return: Constrains around queries
        :rtype: QueryConstrain
        """
        x_min = -0.5
        x_max = 0.5
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)
    
    def result_constrain(self) -> ResultConstrain:
        """
        result_constrain(self) -> ResultConstrain
        | **Description**
        |   See :func:`DataSource.result_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``result_shape``
        |   *Value Range:*

        +-----+-----+-----+------+-----+------+-----+-----+
        |MIN  |a < 0|a = 0|a > 0 |MAX  |a < 0 |a = 0|a > 0|
        +=====+=====+=====+======+=====+======+=====+=====+
        |     |1/4*a|0    |-1/4*a|     |-1/4*a|0    |1/4*a|
        +-----+-----+-----+------+-----+------+-----+-----+

        :return: Constrains around results
        :rtype: ResultConstrain
        """
        y_min = -self.a/4 if self.a>=0 else self.a/4
        y_max = self.a/4 if self.a>=0 else -self.a/4
        result_ranges = np.asarray(tuple((y_min, y_max) for i in range(self.result_shape[0])))
        return ResultConstrain(shape=self.result_shape, ranges=result_ranges)

@dataclass
class DoubleLinearDataSource(DataSource):
    """
    DoubleLinearDataSource(query_shape, result_shape, a, s)
    | **Description**
    |   A ``DoubleLinearDataSource`` is a **function-prior random** source of data choosing one of the following equations at random {``a * x``, ``a * x * s``}. 

    :param query_shape: The expected shape of the queries  (default= (1,))
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results (default= (1,))
    :type result_shape: tuple of ints
    :param a: Coefficient of x (default= 1)
    :type a: float
    :param s: Coefficient that is randomly in- or excluded  (default= 0.5)
    :type s: float
    """
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    a: float = init(default=1)
    s: float = init(default=0.5)

    def query(self, queries):
        """
        query(self, queries) -> data_points
        | **Description**
        |   See :func:`DataSource.query()`

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  
        """
         
        slope = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results_steap = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*self.a) 
        results_flat = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*self.s*self.a)

        results = (1- slope)*results_steap + slope*results_flat

        return queries, results

    def query_constrain(self) -> QueryConstrain:
        """
        query_constrain(self) -> QueryConstrain
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [-0.5, 0.5)

        :return: Constrains around queries
        :rtype: QueryConstrain
        """
        x_min = -0.5
        x_max = 0.5
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)
    
    def result_constrain(self) -> ResultConstrain:
        """
        result_constrain(self) -> ResultConstrain
        | **Description**
        |   See :func:`DataSource.result_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``result_shape``
        |   *Value Range:* 

        +------------+--------+--------+------------+--------+--------+
        |MIN         |a < 0   |a >= 0  |MAX         |a < 0   |a >= 0  |
        +============+========+========+============+========+========+
        |s < -1      |-1/2*a*s|1/2*a*s |s < -1      |1/2*a*s |-1/2*a*s|
        +------------+--------+--------+------------+--------+--------+
        |-1 <= s < 1 |1/2*a   |-1/2*a  |-1 <= s < 1 |-1/2*a  |1/2*a   |
        +------------+--------+--------+------------+--------+--------+
        |1 <= s      |1/2*a*s |-1/2*a*s|1 <= s      |-1/2*a*s|1/2*a*s |
        +------------+--------+--------+------------+--------+--------+

        :return: Constrains around results
        :rtype: ResultConstrain
        """
        y_min = self.s*self.a/2 if (self.a<0 and self.s>1 or self.a>=0 and self.s<-1) else -self.a*self.s/2 if (self.a>=0 and self.s>1 or self.a<0 and self.s<-1) else self.a/2 if (self.a<0 and self.s>=-1 and self.s<=1) else -self.a/2
        y_max = -self.s*self.a/2 if (self.a<0 and self.s>1 or self.a>=0 and self.s<-1) else self.a*self.s/2 if (self.a>=0 and self.s>1 or self.a<0 and self.s<-1) else -self.a/2 if (self.a<0 and self.s>=-1 and self.s<=1) else self.a/2
        result_ranges = np.asarray(tuple((y_min, y_max) for i in range(self.result_shape[0])))
        return ResultConstrain(shape=self.result_shape, ranges=result_ranges)

@dataclass
class HourglassDataSource(DataSource):
    """
    HourglassDataSource(query_shape, result_shape, a)
    | **Description**
    |   A ``HourglassDataSource`` is a **function-prior random** source of data choosing one of the following equations at random {``a * x``, ``-a * x`` , ``-a/2``, ``a/2``}. 

    :param query_shape: The expected shape of the queries (default= (1,))
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results (default= (1,))
    :type result_shape: tuple of ints
    :param a: Coefficient of x (default= 1)
    :type a: float
    """
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    a: float = init(default=1)

    def query(self, queries):
        """
        query(self, queries) -> data_points
        | **Description**
        |   See :func:`DataSource.query()`

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  
        """
         
        kind = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results_up = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*self.a) 
        results_down = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*(-self.a))

        results_dir = (1- kind)*results_up + kind*results_down
        results_const = (1- kind)*0.5 + kind*-0.5

        const = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results = (1- const)*results_dir + const*results_const

        return queries, results

    def query_constrain(self) -> QueryConstrain:
        """
        query_constrain(self) -> QueryConstrain
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [0, 1)

        :return: Constrains around queries
        :rtype: QueryConstrain
        """
        x_min = -0.5
        x_max = 0.5
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)
    
    def result_constrain(self) -> ResultConstrain:
        """
        result_constrain(self) -> ResultConstrain
        | **Description**
        |   See :func:`DataSource.result_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``result_shape``
        |   *Value Range:*

        +---+-----+------+---+------+------+
        |MIN|a < 0|a >= 0|MAX|a < 0 |a >= 0|
        +===+=====+======+===+======+======+
        |   |1/2*a|-1/2*a|   |-1/2*a|1/2*a |
        +---+-----+------+---+------+------+

        :return: Constrains around results
        :rtype: ResultConstrain
        """
        y_min = self.a/2 if self.a<0 else -self.a/2
        y_max = -self.a/2 if self.a<0 else self.a/2
        result_ranges = np.asarray(tuple((y_min, y_max) for i in range(self.result_shape[0])))
        return ResultConstrain(shape=self.result_shape, ranges=result_ranges)

@dataclass
class ZDataSource(DataSource):
    """
    ZDataSource(query_shape, result_shape, a)
    | **Description**
    |   A ``ZDataSource`` is a **function-prior random** source of data choosing one of the following equations at random {``a * x`` , ``-a/2``, ``a/2``}. 

    :param query_shape: The expected shape of the queries (default= (1,))
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results (default= (1,))
    :type result_shape: tuple of ints
    :param a: Coefficient of x (default= 1)
    :type a: float
    """
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    a: float = init(default=1)

    def query(self, queries):
        """
        query(self, queries) -> data_points
        | **Description**
        |   See :func:`DataSource.query()`

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  
        """
         
        kind = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results_const = (1- kind)*0.5 + kind*-0.5
        results_up = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*self.a) 


        const = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results = (1- const)*results_up + const*results_const

        return queries, results

    def query_constrain(self) -> QueryConstrain:
        """
        query_constrain(self) -> QueryConstrain
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [0, 1)

        :return: Constrains around queries
        :rtype: QueryConstrain
        """
        x_min = -0.5
        x_max = 0.5
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)
    
    def result_constrain(self) -> ResultConstrain:
            """
            | **Description**
            |   See :func:`DataSource.result_constrain()` 

            | **Current Constrains**
            |   *Shape:* ``result_shape``
            |   *Value Range:*

            +---+-----+------+---+------+------+
            |MIN|a < 0|a >= 0|MAX|a < 0 |a >= 0|
            +===+=====+======+===+======+======+
            |   |1/2*a|-1/2*a|   |-1/2*a|1/2*a |
            +---+-----+------+---+------+------+

            :return: Constrains around results
            :rtype: ResultConstrain
            """
            y_min = self.a/2 if self.a<0 else -self.a/2
            y_max = -self.a/2 if self.a<0 else self.a/2
            result_ranges = np.asarray(tuple((y_min, y_max) for i in range(self.result_shape[0])))
            return ResultConstrain(shape=self.result_shape, ranges=result_ranges)    

@dataclass
class ZInvDataSource(DataSource):
    """
    ZInvDataSource(query_shape, result_shape, a)
    | **Description**
    |   A ``ZInvDataSource`` is a **function-prior random** source of data choosing one of the following equations at random {``-a * x`` , ``-a/2``, ``a/2``}. 

    :param query_shape: The expected shape of the queries (default= (1,))
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results (default= (1,))
    :type result_shape: tuple of ints
    :param a: Coefficient of x (default= 1)
    :type a: float
    """
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    a: float = init(default=1)

    def query(self, queries):
        """
        query(self, queries) -> data_points
        | **Description**
        |   See :func:`DataSource.query()`

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  
        """
         
        kind = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results_const = (1- kind)*0.5 + kind*-0.5
        results_down = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*(-self.a))

        const = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results = (1- const)*results_down + const*results_const

        return queries, results

    def query_constrain(self) -> QueryConstrain:
        """
        query_constrain(self) -> QueryConstrain
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [0, 1)

        :return: Constrains around queries
        :rtype: QueryConstrain
        """
        x_min = -0.5
        x_max = 0.5
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)
    
    def result_constrain(self) -> ResultConstrain:
        """
        result_constrain(self) -> ResultConstrain
        | **Description**
        |   See :func:`DataSource.result_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``result_shape``
        |   *Value Range:*

        +---+-----+------+---+------+------+
        |MIN|a < 0|a >= 0|MAX|a < 0 |a >= 0|
        +===+=====+======+===+======+======+
        |   |1/2*a|-1/2*a|   |-1/2*a|1/2*a |
        +---+-----+------+---+------+------+

        :return: Constrains around results
        :rtype: ResultConstrain
        """
        y_min = self.a/2 if self.a<0 else -self.a/2
        y_max = -self.a/2 if self.a<0 else self.a/2
        result_ranges = np.asarray(tuple((y_min, y_max) for i in range(self.result_shape[0])))
        return ResultConstrain(shape=self.result_shape, ranges=result_ranges)

@dataclass
class LinearPeriodicDataSource(DataSource):
    """
    LinearPeriodicDataSource(query_shape, result_shape, a, p)
    | **Description**
    |   A ``LinearPeriodicDataSource`` is a **deterministic** source of data representing the equation ``a*x mod p``. 

    :param query_shape: The expected shape of the queries (default= (1,))
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results (default= (1,))
    :type result_shape: tuple of ints
    :param a: Coefficient of x (default= 1)
    :type a: float
    :param p: Modulo divisor (default= 0.2)
    :type p: float
    """
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    a: float = init(default=1)
    p: float = init(default=0.2)


    def query(self, queries):
        """
        query(self, queries) -> data_points
        | **Description**
        |   See :func:`DataSource.query()`

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  
        """
         
        results = np.dot(queries % self.p, np.ones((*self.query_shape, *self.result_shape))*self.a)
        return queries, results

    def query_constrain(self) -> QueryConstrain:
        """
        query_constrain(self) -> QueryConstrain
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [0, 1)

        :return: Constrains around queries
        :rtype: QueryConstrain
        """
        x_min = 0
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)
    
    def result_constrain(self) -> ResultConstrain:
        """
        result_constrain(self) -> ResultConstrain
        | **Description**
        |   See :func:`DataSource.result_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``result_shape``
        |   *Value Range:*

        +-----+-----+-----+-----+-----+-----+-----+-----+
        |MIN  |a < 0|a = 0|a > 0|MAX  |a < 0|a = 0|a > 0|
        +=====+=====+=====+=====+=====+=====+=====+=====+
        |p < 0|p    |0    |p    |p < 0|0    |0    |0    |
        +-----+-----+-----+-----+-----+-----+-----+-----+
        |p = 0|NaN  |NaN  |NaN  |p = 0|NaN  |NaN  |NaN  |
        +-----+-----+-----+-----+-----+-----+-----+-----+
        |p > 0|0    |0    |0    |p > 0|p    |0    |p    |
        +-----+-----+-----+-----+-----+-----+-----+-----+

        :return: Constrains around results
        :rtype: ResultConstrain
        """
        y_min = 0 if self.p>0 or self.p<0 and self.a==0 else self.p if self.p<0 and self.a!=0 else np.nan
        y_max = 0 if self.p<0 or self.p>0 and self.a==0 else self.p if self.p>0 and self.a!=0 else np.nan
        result_ranges = np.asarray(tuple((y_min, y_max) for i in range(self.result_shape[0])))
        return ResultConstrain(shape=self.result_shape, ranges=result_ranges)
    
@dataclass
class LinearStepDataSource(DataSource):
    """
    LinearStepDataSource(query_shape, result_shape, a, p)
    | **Description**
    |   A ``LinearStepDataSource`` is a **deterministic** source of data representing the equation ``a*(x-mod(x,p))/p``. 

    :param query_shape: The expected shape of the queries (default= (1,))
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results (default= (1,))
    :type result_shape: tuple of ints
    :param a: Coefficient of x (default= 1)
    :type a: float
    :param p: Integer divisor (default= 0.2)
    :type p: float
    """
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    a: float = init(default=1)
    p: float = init(default=0.2) 

    def query(self, queries):
        """
        query(self, queries) -> data_points
        | **Description**
        |   See :func:`DataSource.query()`

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  
        """
         
        remainder = queries % self.p
        offset = (queries - remainder) / self.p
        results = np.dot(offset, np.ones((*self.query_shape, *self.result_shape))*self.a)
        return queries, results

    def query_constrain(self) -> QueryConstrain:
        """
        query_constrain(self) -> QueryConstrain
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [0, 1)

        :return: Constrains around queries
        :rtype: QueryConstrain
        """
        x_min = 0
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)
    
    def result_constrain(self) -> ResultConstrain:
        """
        result_constrain(self) -> ResultConstrain
        | **Description**
        |   See :func:`DataSource.result_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``result_shape``
        |   *Value Range:*

        +-----+------------+-----+------------+-----+------------+-----+------------+
        |MIN  |a < 0       |a = 0|a > 0       |MAX  |a < 0       |a = 0|a > 0       |
        +=====+============+=====+============+=====+============+=====+============+
        |p < 0|0           |0    |a*floor(1/p)|p < 0|a*floor(1/p)|0    |0           |
        +-----+------------+-----+------------+-----+------------+-----+------------+
        |p = 0|NaN         |NaN  |NaN         |p = 0|NaN         |NaN  |NaN         |
        +-----+------------+-----+------------+-----+------------+-----+------------+
        |p > 0|a*floor(1/p)|0    |0           |p > 0|0           |0    |a*floor(1/p)|
        +-----+------------+-----+------------+-----+------------+-----+------------+

        :return: Constrains around results
        :rtype: ResultConstrain
        """
        y_min = 0 if self.p>0 and self.a<=0 or self.p<0 and self.a>=0 else self.a*floor(1/self.p) if self.p<0 and self.a>0 or self.p>0 and self.a<0 else np.nan
        y_max = 0 if self.p>0 and self.a>=0 or self.p<0 and self.a<=0 else self.a*floor(1/self.p) if self.p<0 and self.a<0 or self.p>0 and self.a>0 else np.nan
        result_ranges = np.asarray(tuple((y_min, y_max) for i in range(self.result_shape[0])))
        return ResultConstrain(shape=self.result_shape, ranges=result_ranges)

@dataclass
class SineDataSource(DataSource):
    """
    SineDataSource(query_shape, result_shape, a, p)
    | **Description**
    |   A ``SineDataSource`` is a **deterministic** source of data representing the equation ``sin((x-x0)*2pi*p) + y0``. 

    :param query_shape: The expected shape of the queries (default= (1,))
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results (default= (1,))
    :type result_shape: tuple of ints
    :param a: Coefficient of x (default= 1)
    :type a: float
    :param p: Integer divisor (default= 0.2)
    :type p: float
    """
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    a: float = init(default=1)
    p: float = init(default=1)
    y0: float = init(default=0)
    x0: float = init(default=0)


    def query(self, queries):
        """
        query(self, queries) -> data_points
        | **Description**
        |   See :func:`DataSource.query()`

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  
        """
         
        results = np.dot(np.sin((queries-self.x0) * 2 * np.pi * self.p), np.ones((*self.query_shape,*self.result_shape))*self.a) + np.ones(self.result_shape)*self.y0
        return queries, results

    def query_constrain(self) -> QueryConstrain:
        """
        query_constrain(self) -> QueryConstrain
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [0, 1)

        :return: Constrains around queries
        :rtype: QueryConstrain
        """
        x_min = 0
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)

@dataclass
class HypercubeDataSource(DataSource):
    """
    HypercubeDataSource(query_shape, result_shape, w)
    | **Description**
    |   A ``HypercubeDataSource`` is a **function-prior random** source of data choosing for x in [-w,w) one of the following values at random {``-0.5`` , ``0.5``} and else a random value in [-0.5 , 0.5). 

    :param query_shape: The expected shape of the queries (default= (1,))
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results (default= (1,))
    :type result_shape: tuple of ints
    :param w: Outside what range [-w, w) to break down into randomness (default= 0.4)
    :type w: float
    """
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    w: float = init(default=0.4)

    def query(self, queries):
        """
        query(self, queries) -> data_points
        | **Description**
        |   See :func:`DataSource.query()`

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  
        """
         
        kind = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        random = np.random.uniform(-0.5,0.5, size=(queries.shape[0], *self.result_shape))

        results_const = (1- kind)*0.5 + kind*-0.5

        mask = np.all(np.greater(queries, -self.w) * np.less(queries, self.w), axis = 1)[:,None]

        results = mask * results_const + (1 - mask) * random

        return queries, results


    def query_constrain(self) -> QueryConstrain:
        """
        query_constrain(self) -> QueryConstrain
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [-0.5, 0.5)

        :return: Constrains around queries
        :rtype: QueryConstrain
        """
        x_min = -0.5
        x_max = 0.5
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)

@dataclass
class StarDataSource(DataSource):
    """
    StarDataSource(query_shape, result_shape, w)
    | **Description**
    |   A ``StarDataSource`` is a **function-prior random** source of data choosing for x in [-w,w) a random value in [-0.5 , 0.5) and else one of the following equations at random {``-x`` , ``0``, ``x``}. 

    :param query_shape: The expected shape of the queries (default= (1,))
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results (default= (1,))
    :type result_shape: tuple of ints
    :param w: Inside what range [-w, w) to break down into randomness (default= 0.0.05)
    :type w: float
    """
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    w: float = init(default=0.05)

    def query(self, queries):
        """
        query(self, queries) -> data_points
        | **Description**
        |   See :func:`DataSource.query()`

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  
        """
         
        direction = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results_up = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))) 
        results_down = np.dot(queries, -np.ones((*self.query_shape,*self.result_shape)))

        results_dir = (1- direction)*results_up + direction*results_down

        const = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        result_const = const*results_dir

        random = np.random.uniform(-0.5,0.5, size=(queries.shape[0], *self.result_shape))

        mask = np.all(np.greater(queries, -self.w) * np.less(queries, self.w), axis = 1)[:,None]

        results = mask * random + (1 - mask) * result_const

        return queries, results

    def query_constrain(self) -> QueryConstrain:
        """
        query_constrain(self) -> QueryConstrain
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [-0.5, 0.5)

        :return: Constrains around queries
        :rtype: QueryConstrain
        """
        x_min = -0.5
        x_max = 0.5
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)

@dataclass
class HyperSphereDataSource(DataSource):
    """
    HypersphereDataSource(query_shape, result_shape)
    | **Description**
    |   A ``HypersphereDataSource`` is a **function-prior random** source of data choosing one of the following equations at random {``-sqrt(abs(1-x²))``, ``sqrt(abs(1-x²))``}. 

    :param query_shape: The expected shape of the queries (default= (1,))
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results (default= (1,))
    :type result_shape: tuple of ints
    """
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))

    def query(self, queries):
        """
        query(self, queries) -> data_points
        | **Description**
        |   See :func:`DataSource.query()`

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  
        """
         
        x = np.dot(-1*np.square(queries), np.ones((*self.query_shape,*self.result_shape)))
        y = x + np.ones(self.result_shape)
        top_half = np.sqrt(np.abs(y))

        kind = np.random.randint(2,size=(queries.shape[0], *self.result_shape))
        results = top_half * kind + np.negative(top_half) * (1 - kind)
        return queries, results


    def query_constrain(self) -> QueryConstrain:
        """
        query_constrain(self) -> QueryConstrain
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [-1, 1)

        :return: Constrains around queries
        :rtype: QueryConstrain
        """
        x_min = -1
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)

@dataclass
class IndependentDataSource(DataSource):
    """
    IndependentDataSource(reinit, query_shape, result_shape, number_of_distributions, all_distributions, distributions, coefficients)
    | **Description**
    |   An ``IndependentDataSource`` is an **independent** source of data randomly choosing between multiple random distributions to randomly choose from.

    :param reinit: If the DataSource should re-initiate its singleton (default= False)
    :type reinit: bool
    :param query_shape: The expected shape of the queries (default= (1,))
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results (default= (1,))
    :type result_shape: tuple of ints
    :param number_of_distributions: The amount of data distributions to choose from (default= 20)
    :type number_of_distributions: int
    :param all_distributions: A tuple of all differen distributions to choose from (default= (np.random.normal,np.random.uniform,np.random.gamma))
    :type all_distributions: Tuple
    :param distributions: A list of all number_of_distributions many distributions (default= random)
    :type distributions: list
    :param coefficients: The likelihood for each distribution to be chosen (default= random)
    :type coefficients: NDArray[Shape['D'], Number]
    """
    reinit: bool = init(default=False)
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    number_of_distributions: int = init(default=20)

    all_distributions: Tuple = pre_init(default=(np.random.normal,np.random.uniform,np.random.gamma))
    distributions: list = pre_init(default=None)
    coefficients: NDArray[Shape['D'], Number] = pre_init(default=None) # type: ignore

    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Initializes its Singleton.
        |   See :func:`init_singleton` for more.
        """
        super().post_init()
        self.init_singleton()


    def init_singleton(self):
        """
        init_singleton(self) -> None
        | **Description**
        |   Initializes the distributions with random values within the given restrictions.
        """
        if self.distributions is None or self.reinit == True:
            self.distributions = []
            for i in range(self.number_of_distributions):
                loc = np.random.uniform(-10, 10,size=1)
                scale = np.random.uniform(0.1,2,size=1)
                shape: np.ndarray[Any, np.dtype[np.signedinteger[Any]]] = np.random.uniform(0.1,5,size=1) # type: ignore
                distribution = np.random.choice(self.all_distributions)
                if distribution is np.random.normal:
                    self.distributions.append({"type": distribution, "kwargs": {"loc": loc, "scale": scale}})
                elif distribution is np.random.uniform:
                    self.distributions.append({"type": distribution, "kwargs": {"low": loc-scale/2, "high": loc+scale/2}})
                elif distribution is np.random.gamma:
                    self.distributions.append({"type": distribution, "kwargs": {"shape":shape, "scale": scale}})
            coefficients = np.random.uniform(0,1,size=self.number_of_distributions)
            self.coefficients = coefficients / coefficients.sum()


    def query(self, queries):
        """
        query(self, queries) -> data_points
        | **Description**
        |   See :func:`DataSource.query()`

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  
        """
         
        sample_size = queries.shape[0]
        distrs = np.random.choice(a=self.distributions, size=(sample_size, *self.result_shape), p=self.coefficients)
        distrs_flat = distrs.flat
        results_flat = np.empty_like(distrs_flat, dtype=queries.dtype)
        for index, distr in enumerate(distrs_flat):
            results_flat[index] = distr["type"](**distr["kwargs"])
        results = results_flat.reshape((sample_size, *self.result_shape))
        return queries, results

    def query_constrain(self) -> QueryConstrain:
        """
        query_constrain(self) -> QueryConstrain
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [0, 1)

        :return: Constrains around queries
        :rtype: QueryConstrain
        """
        x_min = 0
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)
    
    def __call__(self, **kwargs) -> Self:
        """
        __call__(self, **kwargs) -> IndependentDataSource
        | **Description**
        |   Returns a configured copy of itself.

        :return: A configured copy of itself
        :rtype: IndependentDataSource
        """
        obj = super().__call__( **kwargs)
        obj.distributions = self.distributions
        obj.coefficients = self.coefficients
        return obj

@dataclass
class GaussianProcessDataSource(DataSource):
    """
    GaussianProcessDataSource(reinit, query_shape, result_shape, kern, support_points, min_support, max_support)
    | **Description**
    |   A ``GaussianProcessDataSource`` is a **function-prior random** source of data interpolating between random data points using Gaussian Process Regression.

    :param reinit: If the DataSource should re-initiate its singleton (default= False)
    :type reinit: bool
    :param query_shape: The expected shape of the queries (default= (1,))
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results (default= (1,))
    :type result_shape: tuple of ints
    :param kern: The `kernel <https://gpy.readthedocs.io/en/devel/GPy.kern.html>`_   to use for the gaussian process (default=GPy.Kern.RBF "Radial Basis Function")
    :type kern: GPy.kern.Kern
    :param support_points: The amount of data points to interpolate between in the gaussian process. (default= 2000)
    :type support_points: int
    :param min_support: The lowest permitted query value for each support (default= (-1,))
    :type min_support: tuple of floats
    :param max_support: The highest permitted query value for each support (default= (1,))
    :type max_support: tuple of floats
    """
    reinit: bool = init(default=False)
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    kern: Optional[GPy.kern.Kern] = init(default=None)
    support_points: int= init(default=2000)
    min_support: Tuple[float,...] = init(default=(-1,))
    max_support: Tuple[float,...] = init(default=(1,))
    
    regression: GPy.models.GPRegression = pre_init(default=None)

    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Assigns a RBF Kernel to itself if it has not been assigned one already.
        |   Initializes its Singleton.
        |   See :func:`init_singleton` for more.
        """
        if self.kern is None:
            self.kern = GPy.kern.RBF(input_dim=np.prod(self.query_shape), lengthscale=0.1)
        super().post_init()
        self.init_singleton()
    
    def init_singleton(self):
        """
        init_singleton(self) -> None
        | **Description**
        |   Initializes the Gaussian Process Regression with random values within the given restrictions.
        """
        if self.regression is None or self.reinit == True:
            rng = np.random.RandomState(None)
            support = rng.uniform(self.min_support, self.max_support, (self.support_points, *self.query_shape))

            flat_support = support.reshape((support.shape[0], -1))

            results = np.random.normal(0, 1, (1, *self.result_shape))

            flat_results = results.reshape((1, -1))

            m = GPy.models.GPRegression(flat_support[:1], flat_results, self.kern, noise_var=0.0)

            flat_result = m.posterior_samples_f(flat_support,size=1)[:,:,0]

            self.regression = GPy.models.GPRegression(flat_support, flat_result, self.kern, noise_var=0.0)

    def query(self, queries) -> Tuple[NDArray[Shape["query_nr, ... query_dim"], Number], NDArray[Shape["query_nr, ... result_dim"], Number]]: # type: ignore
        """
        | **Description**
        |   See :func:`DataSource.query()`

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  
        """
         
        flat_queries = queries.reshape((queries.shape[0], -1))
        
        flat_results, pred_cov = self.regression.predict_noiseless(flat_queries)
        results = flat_results.reshape((queries.shape[0], *self.result_shape))

        return queries, results

    def query_constrain(self) -> QueryConstrain:
        """
        query_constrain(self) -> QueryConstrain
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [min_support, max_support)

        :return: Constrains around queries
        :rtype: QueryConstrain
        """
        x_min_max = zip(self.min_support, self.max_support)
        query_ranges = np.asarray(tuple((x_min, x_max) for x_min, x_max in x_min_max))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)


    def __call__(self, **kwargs) -> Self:
        """
        __call__(self, **kwargs) -> GaussianProcessDataSource
        | **Description**
        |   Returns a configured copy of itself.

        :return: A configured copy of itself
        :rtype: GaussianProcessDataSource
        """
        obj: GaussianProcessDataSource = super().__call__( **kwargs)
        obj.regression = self.regression
        return obj # type: ignore
    
@dataclass
class BrownianProcessDataSource(GaussianProcessDataSource):
    """
    BrownianProcessDataSource(reinit, query_shape, result_shape, kern, support_points, min_support, max_support, brown_var)
    | **Description**
    |   A ``BrownianProcessDataSource`` is a **function-prior random** source of data interpolating between random data points using Brownian Motion.

    :param reinit: If the DataSource should re-initiate its singleton (default= False)
    :type reinit: bool
    :param query_shape: The expected shape of the queries (default= (1,))
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results (default= (1,))
    :type result_shape: tuple of ints
    :param kern: The `kernel <https://gpy.readthedocs.io/en/devel/GPy.kern.html>`_   to use for the gaussian process (invariably set to GPy.Kern.Brownian)
    :type kern: GPy.kern.Kern
    :param support_points: The amount of data points to interpolate between in the gaussian process. (default= 2000)
    :type support_points: int
    :param min_support: The lowest permitted query value for each support (default= (0,))
    :type min_support: tuple of floats
    :param max_support: The highest permitted query value for each support (default= (100,))
    :type max_support: tuple of floats
    :param brown_var: The variance of the brownian motion (default= 0.01)
    :type brown_var: float
    """
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    min_support: Tuple[float,...] = init(default=(0,))
    max_support: Tuple[float,...] = init(default=(100,))
    brown_var: float = init(default=0.01)

    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Assigns a Brownian Kernel to itself.
        |   Initializes its Singleton.
        |   See :func:`init_singleton` for more.
        """
        self.kern = GPy.kern.Brownian(variance=self.brown_var)
        super().post_init()
    
@dataclass
class BrownianDriftDataSource(GaussianProcessDataSource):
    """
    BrownianDriftDataSource(reinit, query_shape, result_shape, kern, support_points, brown_var, rbf_var, rbf_leng, min_support, max_support)
    | **Description**
    |   A ``BrownianDriftDataSource`` is a **function-prior random** source of data interpolating between random data points using a linear function y=ab+a where a is a RBF Kernel and b is a Brownian Kernel.

    :param reinit: If the DataSource should re-initiate its singleton (default= False)
    :type reinit: bool
    :param query_shape: The expected shape of the queries (default= (2,))
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results (default= (1,))
    :type result_shape: tuple of ints
    :param kern: The `kernel <https://gpy.readthedocs.io/en/devel/GPy.kern.html>`_   to use for the gaussian process (invariably set to GPy.Kern.Brownian)
    :type kern: GPy.kern.Kern
    :param support_points: The amount of data points to interpolate between in the gaussian process. (default= 2000)
    :type support_points: int
    :param brown_var: The variance of the brownian motion (default= 0.01)
    :type brown_var: float
    :param rbf_var: The variance of the Radial Basis Function (default= 0.25)
    :type rbf_var: float
    :param rbf_leng: The smoothness of the function, where higher values correspond to higher smoothness (default= 0.1)
    :type rbf_leng: float
    :param min_support: The lowest permitted query value for each support (default= (0,))
    :type min_support: tuple of floats
    :param max_support: The highest permitted query value for each support (default= (100,))
    :type max_support: tuple of floats
    """
    query_shape: Tuple[int,...] = init(default=(2,))
    result_shape: Tuple[int,...] = init(default=(1,))
    brown_var: float = init(default=0.01) #0.005
    rbf_var: float = init(default=0.25)
    rbf_leng: float = init(default=0.1) #0.4
    min_support: Tuple[float,...] = init(default=(0,-1))
    max_support: Tuple[float,...] = init(default=(2000,1))

    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Assigns a Brownian_Kernel*RBF_Kernel+RBF_Kernel to itself.
        |   Initializes its Singleton.
        |   See :func:`init_singleton` for more.
        """
        self.kern = GPy.kern.Brownian(active_dims=[0],variance=self.brown_var)*GPy.kern.RBF(input_dim=1, lengthscale=self.rbf_leng, variance=self.rbf_var, active_dims=[1])+GPy.kern.RBF(input_dim=1, lengthscale=self.rbf_leng, variance=self.rbf_var, active_dims=[1])
        super().post_init()

@dataclass
class RBFDriftDataSource(GaussianProcessDataSource):
    """
    RBFDriftDataSource(reinit, query_shape, result_shape, kern, support_points, brown_var, rbf_var, rbf_leng, min_support, max_support)
    | **Description**
    |   A ``RBFDriftDataSource`` is a **function-prior random** source of data interpolating with a RBF kernel with RBF drift.

    :param reinit: If the DataSource should re-initiate its singleton (default= False)
    :type reinit: bool
    :param query_shape: The expected shape of the queries (default= (2,))
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results (default= (1,))
    :type result_shape: tuple of ints
    :param kern: The `kernel <https://gpy.readthedocs.io/en/devel/GPy.kern.html>`_   to use for the gaussian process (invariably set to GPy.Kern.Brownian)
    :type kern: GPy.kern.Kern
    :param support_points: The amount of data points to interpolate between in the gaussian process. (default= 2000)
    :type support_points: int
    :param brown_var: The variance of the brownian motion (default= 0.01)
    :type brown_var: float
    :param rbf_var: The variance of the Radial Basis Function (default= 0.25)
    :type rbf_var: float
    :param rbf_leng: The smoothness of the function, where higher values correspond to higher smoothness (default= 0.1)
    :type rbf_leng: float
    :param min_support: The lowest permitted query value for each support (default= (0, -1))
    :type min_support: tuple of floats
    :param max_support: The highest permitted query value for each support (default= (2000, 1))
    :type max_support: tuple of floats
    """
    query_shape: Tuple[int,...] = init(default=(2,))
    result_shape: Tuple[int,...] = init(default=(1,))
    brown_var: float = init(default=0.01) #0.005
    rbf_var: float = init(default=0.25)
    rbf_leng: float = init(default=0.1) #0.4
    min_support: Tuple[float,...] = init(default=(0,-1))
    max_support: Tuple[float,...] = init(default=(2000,1))

    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Initializes its Kernel.
        |   Initializes its Singleton.
        |   See :func:`init_singleton` for more.
        """
        self.kern = GPy.kern.RBF(input_dim=1, active_dims=[0],variance=self.rbf_var*2, lengthscale=self.brown_var*2000)*GPy.kern.RBF(input_dim=1, lengthscale=self.rbf_leng, variance=self.rbf_var, active_dims=[1])+GPy.kern.RBF(input_dim=1, lengthscale=self.rbf_leng, variance=self.rbf_var, active_dims=[1])
        super().post_init()

@dataclass
class SinDriftDataSource(GaussianProcessDataSource):
    """
    SinDriftDataSource(reinit, query_shape, result_shape, kern, support_points, brown_var, rbf_var, rbf_leng, min_support, max_support)
    | **Description**
    |   A ``SinDriftDataSource`` is a **function-prior random** source of data interpolating data points with a cosine kernel with rbf drift.

    :param reinit: If the DataSource should re-initiate its singleton (default= False)
    :type reinit: bool
    :param query_shape: The expected shape of the queries (default= (2,))
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results (default= (1,))
    :type result_shape: tuple of ints
    :param kern: The `kernel <https://gpy.readthedocs.io/en/devel/GPy.kern.html>`_   to use for the gaussian process (invariably set to GPy.Kern.Brownian)
    :type kern: GPy.kern.Kern
    :param support_points: The amount of data points to interpolate between in the gaussian process. (default= 2000)
    :type support_points: int
    :param brown_var: The variance of the brownian motion (default= 0.01)
    :type brown_var: float
    :param rbf_var: The variance of the Radial Basis Function (default= 0.25)
    :type rbf_var: float
    :param rbf_leng: The smoothness of the function, where higher values correspond to higher smoothness (default= 0.1)
    :type rbf_leng: float
    :param min_support: The lowest permitted query value for each support (default= (0, -1))
    :type min_support: tuple of floats
    :param max_support: The highest permitted query value for each support (default= (2000, 1))
    :type max_support: tuple of floats
    """
    query_shape: Tuple[int,...] = init(default=(2,))
    result_shape: Tuple[int,...] = init(default=(1,))
    brown_var: float = init(default=0.005) #0.005
    rbf_var: float = init(default=0.25)
    rbf_leng: float = init(default=0.1) #0.4
    min_support: Tuple[float,...] = init(default=(0,-1))
    max_support: Tuple[float,...] = init(default=(2000,1))


    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Initializes its Kernel.
        |   Initializes its Singleton.
        |   See :func:`init_singleton` for more.
        """
        self.kern = GPy.kern.Cosine(input_dim=1, active_dims=[0],variance=self.rbf_var*2, lengthscale=self.brown_var*2000)*GPy.kern.RBF(input_dim=1, lengthscale=self.rbf_leng, variance=self.rbf_var, active_dims=[1])+GPy.kern.RBF(input_dim=1, lengthscale=self.rbf_leng, variance=self.rbf_var, active_dims=[1])
        super().post_init()

@dataclass
class MixedDriftDataSource(GaussianProcessDataSource):
    """
    MixedDriftDataSource(support_points, reinit, query_shape, result_shape, brown_var, rbf_var, rbf_leng, min_support, max_support)
    | **Description**
    |   A ``MixedDriftDataSource`` is a **function-prior random** source of data interpolating data points with a linear combination of RBF, Brownian and Cosine kernels and RBF drift.

    :param support_points: The amount of data points to interpolate between in the gaussian process. (default= 2000)
    :type support_points: int
    :param reinit: If the DataSource should re-initiate its singleton (default= False)
    :type reinit: bool
    :param query_shape: The expected shape of the queries (default= (2,))
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results (default= (1,))
    :type result_shape: tuple of ints
    :param brown_var: The variance of the brownian motion (default= 0.01)
    :type brown_var: float
    :param rbf_var: The variance of the Radial Basis Function (default= 0.25)
    :type rbf_var: float
    :param rbf_leng: The smoothness of the function, where higher values correspond to higher smoothness (default= 0.1)
    :type rbf_leng: float
    :param min_support: The lowest permitted query value for each support (default= (0, -1))
    :type min_support: tuple of floats
    :param max_support: The highest permitted query value for each support (default= (2000, 1))
    :type max_support: tuple of floats
    """
    support_points: int= init(default=2000)
    reinit: bool = init(default=False)
    query_shape: Tuple[int,...] = init(default=(2,))
    result_shape: Tuple[int,...] = init(default=(1,))
    brown_var: float = init(default=0.01) #0.005
    rbf_var: float = init(default=0.25)
    rbf_leng: float = init(default=0.1) #0.4
    min_support: Tuple[float,...] = init(default=(0,-1))
    max_support: Tuple[float,...] = init(default=(2000,1))

    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Initializes its 7 kernels.
        |   Initializes its Singleton.
        |   See :func:`init_singleton` for more.
        """
        self.gp_i = GaussianProcessDataSource(
            kern=GPy.kern.RBF(input_dim=1, lengthscale=self.rbf_leng, variance=self.rbf_var, active_dims=[0]),
            reinit=self.reinit, support_points=self.support_points, min_support=(self.min_support[1],),
            max_support=(self.max_support[1],),
            )()
        self.gp_w1 = GaussianProcessDataSource(
            kern=GPy.kern.RBF(input_dim=1, lengthscale=self.rbf_leng, variance=self.rbf_var, active_dims=[0]),
            reinit=self.reinit, support_points=self.support_points, min_support=(self.min_support[1],),
            max_support=(self.max_support[1],),
            )()
        self.gp_w2 = GaussianProcessDataSource(
            kern=GPy.kern.RBF(input_dim=1, lengthscale=self.rbf_leng, variance=self.rbf_var, active_dims=[0]),
            reinit=self.reinit, support_points=self.support_points, min_support=(self.min_support[1],),
            max_support=(self.max_support[1],),
            )()
        self.gp_w3 = GaussianProcessDataSource(
            kern=GPy.kern.RBF(input_dim=1, lengthscale=self.rbf_leng, variance=self.rbf_var, active_dims=[0]),
            reinit=self.reinit, support_points=self.support_points, min_support=(self.min_support[1],),
            max_support=(self.max_support[1],),
            )()
        self.gp_b1 = GaussianProcessDataSource(
            kern=GPy.kern.Brownian(active_dims=[0],variance=self.brown_var),
            reinit=self.reinit, support_points=self.support_points, min_support=(self.min_support[0],),
            max_support=(self.max_support[0],),
            )()
        self.gp_b2 = GaussianProcessDataSource(
            kern=GPy.kern.RBF(input_dim=1, active_dims=[0],variance=self.rbf_var*2, lengthscale=self.brown_var*2000),
            reinit=self.reinit, support_points=self.support_points, min_support=(self.min_support[0],),
            max_support=(self.max_support[0],),
            )()
        self.gp_b3 = GaussianProcessDataSource(
            kern=GPy.kern.Cosine(input_dim=1, active_dims=[0],variance=self.rbf_var*2, lengthscale=self.brown_var*2000),
            reinit=self.reinit, support_points=self.support_points, min_support=(self.min_support[0],),
            max_support=(self.max_support[0],),
            )()
        super().post_init()

    def query(self, queries):
        """
        query(self, queries) -> data_points
        | **Description**
        |   See :func:`DataSource.query()`

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  
        """
         
        flat_queries = queries.reshape((queries.shape[0], -1))

        y_i = self.gp_i.query(flat_queries[:,1:])[1]

        y_w1 = self.gp_w1.query(flat_queries[:,1:])[1]
        y_w2 = self.gp_w2.query(flat_queries[:,1:])[1]
        y_w3 = self.gp_w3.query(flat_queries[:,1:])[1]

        y_b1 = self.gp_b1.query(flat_queries[:,:1])[1]
        y_b2 = self.gp_b2.query(flat_queries[:,:1])[1]
        y_b3 = self.gp_b3.query(flat_queries[:,:1])[1]

        
        flat_results = y_i + y_w1*y_b1 + y_w2*y_b2 + y_w3*y_b3
        results = flat_results.reshape((queries.shape[0], *self.result_shape))

        return queries, results

    def query_constrain(self) -> QueryConstrain:
        """
        query_constrain(self) -> QueryConstrain
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [min_support, max_support)

        :return: Constrains around queries
        :rtype: QueryConstrain
        """
        x_min_max = zip(self.min_support, self.max_support)
        query_ranges = np.asarray(tuple((x_min, x_max) for x_min, x_max in x_min_max))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)

@dataclass
class MixedBrownDriftDataSource(GaussianProcessDataSource):
    """
    MixedDriftDataSource(support_points, reinit, query_shape, result_shape, brown_var, rbf_var, rbf_leng, min_support, max_support)
    | **Description**
    |   A ``MixedDriftDataSource`` is a **function-prior random** source of data interpolating data points with a linear combination of RBF, Brownian and Cosine kernels and Brownian Drift.

    :param support_points: The amount of data points to interpolate between in the gaussian process. (default= 2000)
    :type support_points: int
    :param reinit: If the DataSource should re-initiate its singleton (default= False)
    :type reinit: bool
    :param query_shape: The expected shape of the queries (default= (2,))
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results (default= (1,))
    :type result_shape: tuple of ints
    :param brown_var: The variance of the brownian motion (default= 0.01)
    :type brown_var: float
    :param rbf_var: The variance of the Radial Basis Function (default= 0.25)
    :type rbf_var: float
    :param rbf_leng: The smoothness of the function, where higher values correspond to higher smoothness (default= 0.1)
    :type rbf_leng: float
    :param min_support: The lowest permitted query value for each support (default= (0, -1))
    :type min_support: tuple of floats
    :param max_support: The highest permitted query value for each support (default= (2000, 1))
    :type max_support: tuple of floats
    """
    support_points: int= init(default=2000)
    reinit: bool = init(default=False)
    query_shape: Tuple[int,...] = init(default=(2,))
    result_shape: Tuple[int,...] = init(default=(1,))
    brown_var: float = init(default=0.01) #0.005
    rbf_var: float = init(default=0.25)
    rbf_leng: float = init(default=0.1) #0.4
    min_support: Tuple[float,...] = init(default=(0,-1))
    max_support: Tuple[float,...] = init(default=(2000,1))

    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Initializes its 7 kernels.
        |   Initializes its Singleton.
        |   See :func:`init_singleton` for more.
        """
        self.gp_i = GaussianProcessDataSource(
            kern=GPy.kern.RBF(input_dim=1, lengthscale=self.rbf_leng, variance=self.rbf_var, active_dims=[0]),
            reinit=self.reinit, support_points=self.support_points, min_support=(self.min_support[1],),
            max_support=(self.max_support[1],),
            )()
        self.gp_w1 = GaussianProcessDataSource(
            kern=GPy.kern.RBF(input_dim=1, lengthscale=self.rbf_leng, variance=self.rbf_var, active_dims=[0]),
            reinit=self.reinit, support_points=self.support_points, min_support=(self.min_support[1],),
            max_support=(self.max_support[1],),
            )()
        self.gp_w2 = GaussianProcessDataSource(
            kern=GPy.kern.RBF(input_dim=1, lengthscale=self.rbf_leng, variance=self.rbf_var, active_dims=[0]),
            reinit=self.reinit, support_points=self.support_points, min_support=(self.min_support[1],),
            max_support=(self.max_support[1],),
            )()
        self.gp_w3 = GaussianProcessDataSource(
            kern=GPy.kern.RBF(input_dim=1, lengthscale=self.rbf_leng, variance=self.rbf_var, active_dims=[0]),
            reinit=self.reinit, support_points=self.support_points, min_support=(self.min_support[1],),
            max_support=(self.max_support[1],),
            )()
        self.gp_b1 = GaussianProcessDataSource(
            kern=GPy.kern.Brownian(active_dims=[0],variance=self.brown_var),
            reinit=self.reinit, support_points=self.support_points, min_support=(self.min_support[0],),
            max_support=(self.max_support[0],),
            )()
        self.gp_b2 = GaussianProcessDataSource(
            kern=GPy.kern.Brownian(active_dims=[0],variance=self.brown_var),
            reinit=self.reinit, support_points=self.support_points, min_support=(self.min_support[0],),
            max_support=(self.max_support[0],),
            )()
        self.gp_b3 = GaussianProcessDataSource(
            kern=GPy.kern.Brownian(active_dims=[0],variance=self.brown_var),
            reinit=self.reinit, support_points=self.support_points, min_support=(self.min_support[0],),
            max_support=(self.max_support[0],),
            )()
        super().post_init()

    def query(self, queries):
        """
        query(self, queries) -> data_points
        | **Description**
        |   See :func:`DataSource.query()`

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  
        """
         
        flat_queries = queries.reshape((queries.shape[0], -1))

        y_i = self.gp_i.query(flat_queries[:,1:])[1]

        y_w1 = self.gp_w1.query(flat_queries[:,1:])[1]
        y_w2 = self.gp_w2.query(flat_queries[:,1:])[1]
        y_w3 = self.gp_w3.query(flat_queries[:,1:])[1]

        y_b1 = self.gp_b1.query(flat_queries[:,:1])[1]
        y_b2 = self.gp_b2.query(flat_queries[:,:1])[1]
        y_b3 = self.gp_b3.query(flat_queries[:,:1])[1]

        
        flat_results = y_i + y_w1*y_b1 + y_w2*y_b2 + y_w3*y_b3
        results = flat_results.reshape((queries.shape[0], *self.result_shape))

        return queries, results

    def query_constrain(self) -> QueryConstrain:
        """
        query_constrain(self) -> QueryConstrain
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [min_support, max_support)

        :return: Constrains around queries
        :rtype: QueryConstrain
        """
        x_min_max = zip(self.min_support, self.max_support)
        query_ranges = np.asarray(tuple((x_min, x_max) for x_min, x_max in x_min_max))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)

@dataclass
class TimeBehaviorDataSource(TimeDataSource):
    """
    TimeBehaviorDataSource(query_shape, result_shape, behavior)
    | **Description**
    |   A ``TimeBehaviorDataSource`` is an **independent** source of data depending on the DataBehavior over time. 

    :param result_shape: The expected shape of the results (default= (1,))
    :type result_shape: tuple of ints
    :param behavior: How the data behaves over time
    :type behavior: DataBehavior
    """
    result_shape: Tuple[int,...] = init(default=(1,))
    behavior: DataBehavior = init()
    change_times: NDArray[Shape["change_times"], Number] = post_init() # type: ignore
    change_values: NDArray[Shape["change_values"], Number] = post_init() # type: ignore
    current_time: float = pre_init(default=0)

    def post_init(self):
        """
        post_init(self) -> None
        | **Description**
        |   Initializes its Singleton.
        |   See :func:`init_singleton` for more.
        """
        super().post_init()
        self.behavior = is_set(self.behavior)(query_constrain=self.query_constrain()) # type: ignore
        self.change_times, self.change_values = self.behavior.behavior()


    @property
    def exhausted(self):
        """
        exhausted(self) -> bool
        | **Description**
        |   True if current time has exceeded the data source's stop time.

        :return: Exhausted or not
        :rtype: bool
        """
        return self.current_time < self.behavior.stop_time

    def query(self, queries: NDArray[ Shape["query_nr, ... query_dim"], Number]) -> Tuple[NDArray[Shape["query_nr, ... query_dim"], Number], NDArray[Shape["query_nr, ... result_dim"], Number]]: # type: ignore
        """
        | **Description**
        |   See :func:`DataSource.query()`

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  
        """
         
        times = queries
        self.current_time = times[-1,0]

        indices = np.searchsorted(self.change_times, times[...,0],side='right') -1

        results = self.change_values[indices][:,None]

        return queries, results
    
    def query_constrain(self) -> QueryConstrain:
        """
        query_constrain(self) -> QueryConstrain
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constrains**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [behavior.start_time, behavior.stop_time)

        :return: Constrains around queries
        :rtype: QueryConstrain
        """
        return QueryConstrain(count=None, shape=self.query_shape, ranges=np.asarray(((self.behavior.start_time, self.behavior.stop_time),)))