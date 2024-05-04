"""
:doc:`Core Module </core/oracle/data_source>`
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from dataclasses import dataclass, field, InitVar

import numpy as np
import GPy

from alts.core.oracle.data_source import DataSource, TimeDataSource
from alts.core.data.constrains import QueryConstrain

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
    | **Description**
    |   A ``RandomUniformDataSource`` is a random source of data.
    |   For more details see `numpy.random.uniorm <https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html>`_.

    :param query_shape: The expected shape of the queries
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results
    :type result_shape: tuple of ints
    :param u: The upper bound of query values (exclusive)
    :type u: float
    :param l: The lower bound of query values (inclusive)
    :type l: float
    """
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    u: float = init(default=1)
    l: float = init(default=0)


    def query(self, queries):
        """
        | **Description**
        |   ``query()`` is the access point to the data of the ``DataSource``.
        |   It returns *results* in the range of ``[l,u)`` to given *queries* in the range of ``[0,1)`` upon request.

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  

        | **Example**
        |   Let ``y = query(x)``, ``l = 0.0`` and ``u = 1.0``. Since  ``l <= y < u``, we now have ``0.0 <= y < 1.0``.
        |   The resulting graph of ``y = query(x)`` might look something like this.
        """
        results = np.random.uniform(low=self.l, high=self.u, size=(queries.shape[0], * self.result_shape))
        return queries, results

    def query_constrain(self) -> QueryConstrain:
        """
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constraints**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [0, 1)

        :return: Constraints around queries
        :rtype: QueryConstrain
        """
        x_min = 0
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)


@dataclass
class LineDataSource(DataSource):
    """
    | **Description**
    |   A ``LineDataSource`` is a deterministic source of data representing a linear equation ``y = ax + b``.

    :param query_shape: The expected shape of the queries
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results
    :type result_shape: tuple of ints
    :param a: Coefficient of degree 1
    :type a: float
    :param b: Coefficient of degree 0
    :type b: float
    """
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    a: float = init(default=1)
    b: float = init(default=0)


    def query(self, queries):
        """
        | **Description**
        |   ``query()`` is the access point to the data of the ``DataSource``.
        |   It returns *results* in the range of TODO to given *queries* in the range of ``[0,1)`` upon request.

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  

        | **Example**
        |   Let ``y = query(x)``, ``a = 1.0`` and ``b = 0.0``. This represents the linear equation ``y = x``.
        |   The resulting graph of ``y = query(x)`` should look something like this.
        """
        results = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*self.a) + np.ones(self.result_shape)*self.b
        return queries, results

    def query_constrain(self) -> QueryConstrain:
        """
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constraints**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [0, 1)

        :return: Constraints around queries
        :rtype: QueryConstrain
        """
        x_min = 0
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)



@dataclass
class SquareDataSource(DataSource):
    """
    | **Description**
    |   A ``SquareDataSource`` is a deterministic source of data representing a square parabola ``s * (x - x0)^2 + y0``. 

    :param query_shape: The expected shape of the queries
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results
    :type result_shape: tuple of ints
    :param x0: Offset of the parabola in x-direction
    :type x0: float
    :param y0: Offset of the parabola in y-direction
    :type y0: float
    :param s: Coefficient of degree 2
    :type s: float
    """
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    x0: float = init(default=0.5)
    y0: float = init(default=0)
    s: float = init(default=5)

    def query(self, queries):
        """
        | **Description**
        |   ``query()`` is the access point to the data of the ``DataSource``.
        |   It returns *results* in the range of TODO to given *queries* in the range of ``[0,1)`` upon request.

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  

        | **Example**
        |   Let ``y = query(x)``, ``x0 = 0.5``, ``y0 = 0.0`` and ``s = 5.0``. This represents the quadratic equation ``y = 5 * (x - 0.5)^2 + 0``.
        |   The resulting graph of ``y = query(x)`` should look something like this.
        """
        results = np.dot((queries - self.x0)**2, np.ones((*self.query_shape,*self.result_shape))*self.s) + np.ones(self.result_shape)*self.y0
        return queries, results

    def query_constrain(self) -> QueryConstrain:
        """
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constraints**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [0, 1)

        :return: Constraints around queries
        :rtype: QueryConstrain
        """
        x_min = 0
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)

@dataclass
class PowDataSource(DataSource):
    """
    | **Description**
    |   A ``PowDataSource`` is a deterministic source of data representing an exponential equation ``s * x^power``. 

    :param query_shape: The expected shape of the queries
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results
    :type result_shape: tuple of ints
    :param power: Power of x
    :type power: float
    :param s: Coefficient of x^power
    :type s: float
    """
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    power: float = init(default=3)
    s: float = init(default=1)

    def query(self, queries):
        """
        | **Description**
        |   ``query()`` is the access point to the data of the ``DataSource``.
        |   It returns *results* in the range of TODO to given *queries* in the range of ``[0,1)`` upon request.

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  

        | **Example**
        |   Let ``y = query(x)``, ``power = 3.0`` and ``s = 1.0``. This represents the exponential equation ``y = 1 * x^3``.
        |   The resulting graph of ``y = query(x)`` should look something like this.   
        """
        results = np.dot(np.power(queries, self.power), np.ones((*self.query_shape,*self.result_shape))*self.s)
        return queries, results

    def query_constrain(self) -> QueryConstrain:
        """
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constraints**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [0, 1)

        :return: Constraints around queries
        :rtype: QueryConstrain
        """
        x_min = 0
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)

@dataclass
class ExpDataSource(DataSource):
    """
    | **Description**
    |   An ``ExpDataSource`` is a deterministic source of data representing an exponential equation ``s * base^x``. 

    :param query_shape: The expected shape of the queries
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results
    :type result_shape: tuple of ints
    :param base: Basis to the exponent x
    :type base: float
    :param s: Coefficient of base^x
    :type s: float
    """
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    base: float = init(default=2)
    s: float = init(default=1)

    def query(self, queries):
        """
        | **Description**
        |   ``query()`` is the access point to the data of the ``DataSource``.
        |   It returns *results* in the range of TODO to given *queries* in the range of ``[0,1)`` upon request.

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  

        | **Example**
        |   Let ``y = query(x)``, ``base = 2.0`` and ``s = 1.0``. This represents the exponential equation ``y = 1 * 2^x``.
        |   The resulting graph of ``y = query(x)`` should look something like this.     
        """
        results = np.dot(np.power(self.base, queries*self.s), np.ones((*self.query_shape,*self.result_shape)))
        return queries, results

    def query_constrain(self) -> QueryConstrain:
        """
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constraints**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [0, 1)

        :return: Constraints around queries
        :rtype: QueryConstrain
        """
        x_min = 0
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)

    
@dataclass
class InterpolatingDataSource(DataSource):
    """
    | **Description**
    |   TODO
    """
    data_sampler: DataSampler = init()
    interpolation_strategy: InterpolationStrategy = init()

    def post_init(self):
        super().post_init()
        self.data_sampler = self.data_sampler()
        self.interpolation_strategy = self.interpolation_strategy(self.data_sampler)

    def query(self, queries):
        data_points = self.data_sampler.query(queries)
        data_points = self.interpolation_strategy.interpolate(data_points)
        return data_points
    
    def query_constrain(self) -> QueryConstrain:
        return self.interpolation_strategy.query_constrain()



@dataclass
class CrossDataSource(DataSource):
    """
    | **Description**
    |   A ``CrossDataSource`` is a semi-random source of data choosing one of the following equations at random {``-a * x``, ``a * x``}. 

    :param query_shape: The expected shape of the queries
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results
    :type result_shape: tuple of ints
    :param a: Coefficient of x
    :type a: float
    """
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    a: float = init(default=1)

    def query(self, queries):
        """
        | **Description**
        |   ``query()`` is the access point to the data of the ``DataSource``.
        |   It returns *results* in the range of TODO to given *queries* in the range of ``[-0.5,0.5)`` upon request.

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  

        | **Example**
        |   Let ``y = query(x)`` and ``a = 1.0``. This represents the linear equations {``y = -1 * x``, ``y = 1 * x``}.
        |   Upon each call of query() it will choose one of those two equations at random to evaluate.
        |   The resulting graph of ``y = query(x)`` should look something like this. 
        """
        direction = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results_up = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*self.a) 
        results_down = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*(-self.a))

        results = (1- direction)*results_up + direction*results_down

        return queries, results

    def query_constrain(self) -> QueryConstrain:
        """
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constraints**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [-0.5, 0.5)

        :return: Constraints around queries
        :rtype: QueryConstrain
        """
        x_min = -0.5
        x_max = 0.5
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)

@dataclass
class DoubleLinearDataSource(DataSource):
    """
    | **Description**
    |   A ``DoubleLinearDataSource`` is a semi-random source of data choosing one of the following equations at random {``a * x``, ``a * x * slope_factor``}. 

    :param query_shape: The expected shape of the queries
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results
    :type result_shape: tuple of ints
    :param a: Coefficient of x
    :type a: float
    :param slope_factor: Coefficient that is randomly in- or excluded 
    :type slope_factor: float
    """
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    a: float = init(default=1)
    slope_factor: float = init(default=0.5)

    def query(self, queries):
        """
        | **Description**
        |   ``query()`` is the access point to the data of the ``DataSource``.
        |   It returns *results* in the range of TODO to given *queries* in the range of ``[-0.5,0.5)`` upon request.

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  

        | **Example**
        |   Let ``y = query(x)``, ``a = 1.0`` and ``slope_factor = 0.5``. This represents the linear equations {``y = 1 * x``, ``y = 0.5 * 1 * x``}.
        |   Upon each call of query() it will choose one of those two equations at random to evaluate.
        |   The resulting graph of ``y = query(x)`` should look something like this.  
        """
        slope = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results_steap = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*self.a) 
        results_flat = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*self.slope_factor*self.a)

        results = (1- slope)*results_steap + slope*results_flat

        return queries, results

    def query_constrain(self) -> QueryConstrain:
        """
        | **Description**
        |   See :func:`DataSource.query_constrain()` 

        | **Current Constraints**
        |   *Shape:* ``query_shape``
        |   *Value Range:* [-0.5, 0.5)

        :return: Constraints around queries
        :rtype: QueryConstrain
        """
        x_min = -0.5
        x_max = 0.5
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)

        
@dataclass
class HourglassDataSource(DataSource):
    """
    TODO
    """
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    a: float = init(default=1)

    def query(self, queries):

        kind = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results_up = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*self.a) 
        results_down = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*(-self.a))

        results_dir = (1- kind)*results_up + kind*results_down
        results_const = (1- kind)*0.5 + kind*-0.5

        const = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results = (1- const)*results_dir + const*results_const

        return queries, results

    def query_constrain(self) -> QueryConstrain:
        x_min = -0.5
        x_max = 0.5
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)

@dataclass
class ZDataSource(DataSource):

    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    a: float = init(default=1)

    def query(self, queries):

        kind = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results_const = (1- kind)*0.5 + kind*-0.5
        results_up = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*self.a) 


        const = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results = (1- const)*results_up + const*results_const

        return queries, results

    def query_constrain(self) -> QueryConstrain:
        x_min = -0.5
        x_max = 0.5
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)

@dataclass
class ZInvDataSource(DataSource):

    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    a: float = init(default=1)

    def query(self, queries):

        kind = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results_const = (1- kind)*0.5 + kind*-0.5
        results_down = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*(-self.a))

        const = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results = (1- const)*results_down + const*results_const

        return queries, results

    def query_constrain(self) -> QueryConstrain:
        x_min = -0.5
        x_max = 0.5
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)

@dataclass
class LinearPeriodicDataSource(DataSource):

    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    a: float = init(default=1)
    p: float = init(default=0.2)


    def query(self, queries):
        results = np.dot(queries % self.p, np.ones((*self.query_shape, *self.result_shape))*self.a)
        return queries, results

    def query_constrain(self) -> QueryConstrain:
        x_min = 0
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)
    
@dataclass
class LinearStepDataSource(DataSource):

    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    a: float = init(default=1)
    p: float = init(default=0.2)

    def query(self, queries):
        remainder = queries % self.p
        offset = (queries - remainder) / self.p
        results = np.dot(offset, np.ones((*self.query_shape, *self.result_shape))*self.a)
        return queries, results

    def query_constrain(self) -> QueryConstrain:
        x_min = 0
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)

@dataclass
class SineDataSource(DataSource):

    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    a: float = init(default=1)
    p: float = init(default=1)
    y0: float = init(default=0)
    x0: float = init(default=0)


    def query(self, queries):
        results = np.dot(np.sin((queries-self.x0) * 2 * np.pi * self.p), np.ones((*self.query_shape,*self.result_shape))*self.a) + np.ones(self.result_shape)*self.y0
        return queries, results

    def query_constrain(self) -> QueryConstrain:
        x_min = 0
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)


@dataclass
class HypercubeDataSource(DataSource):

    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    w: float = init(default=0.4)

    def query(self, queries):

        kind = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        random = np.random.uniform(-0.5,0.5, size=(queries.shape[0], *self.result_shape))

        results_const = (1- kind)*0.5 + kind*-0.5

        mask = np.all(np.greater(queries, -self.w) * np.less(queries, self.w), axis = 1)[:,None]

        results = mask * results_const + (1 - mask) * random

        return queries, results


    def query_constrain(self) -> QueryConstrain:
        x_min = -0.5
        x_max = 0.5
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)

@dataclass
class StarDataSource(DataSource):

    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    w: float = init(default=0.05)

    def query(self, queries):

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
        x_min = -0.5
        x_max = 0.5
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)


@dataclass
class HyperSphereDataSource(DataSource):

    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))

    def query(self, queries):
        x = np.dot(-1*np.square(queries), np.ones((*self.query_shape,*self.result_shape)))
        y = x + np.ones(self.result_shape)
        top_half = np.sqrt(np.abs(y))

        kind = np.random.randint(2,size=(queries.shape[0], *self.result_shape))
        results = top_half * kind + np.negative(top_half) * (1 - kind)
        return queries, results


    def query_constrain(self) -> QueryConstrain:
        x_min = -1
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)

@dataclass
class IndependentDataSource(DataSource):
    reinit: bool = init(default=False)
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    number_of_distributions: int = init(default=20)

    all_distributions: Tuple = pre_init(default=(np.random.normal,np.random.uniform,np.random.gamma))
    distributions: list = pre_init(default=None)
    coefficients: NDArray[Shape['D'], Number] = pre_init(default=None)

    def post_init(self):
        super().post_init()
        self.init_singleton()


    def init_singleton(self):
        if self.distributions is None or self.reinit == True:
            self.distributions = []
            for i in range(self.number_of_distributions):
                loc = np.random.uniform(-10, 10,size=1)
                scale = np.random.uniform(0.1,2,size=1)
                shape: ndarray[Any, dtype[signedinteger[Any]]] = np.random.uniform(0.1,5,size=1)
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
        sample_size = queries.shape[0]
        distrs = np.random.choice(a=self.distributions, size=(sample_size, *self.result_shape), p=self.coefficients)
        distrs_flat = distrs.flat
        results_flat = np.empty_like(distrs_flat, dtype=queries.dtype)
        for index, distr in enumerate(distrs_flat):
            results_flat[index] = distr["type"](**distr["kwargs"])
        results = results_flat.reshape((sample_size, *self.result_shape))
        return queries, results

    def query_constrain(self) -> QueryConstrain:
        x_min = 0
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)
    
    def __call__(self, **kwargs) -> Self:
        obj = super().__call__( **kwargs)
        obj.distributions = self.distributions
        obj.coefficients = self.coefficients
        return obj

@dataclass
class GaussianProcessDataSource(DataSource):

    reinit: bool = init(default=False)
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    kern: Optional[GPy.kern.Kern] = init(default=None)
    support_points: int= init(default=2000)
    min_support: Tuple[float,...] = init(default=(-1,))
    max_support: Tuple[float,...] = init(default=(1,))
    
    regression: GPy.models.GPRegression = pre_init(default=None)

    def post_init(self):
        if self.kern is None:
            self.kern = GPy.kern.RBF(input_dim=np.prod(self.query_shape), lengthscale=0.1)
        super().post_init()
        self.init_singleton()
    
    def init_singleton(self):
        if self.regression is None or self.reinit == True:
            rng = np.random.RandomState(None)
            support = rng.uniform(self.min_support, self.max_support, (self.support_points, *self.query_shape))

            flat_support = support.reshape((support.shape[0], -1))

            results = np.random.normal(0, 1, (1, *self.result_shape))

            flat_results = results.reshape((1, -1))

            m = GPy.models.GPRegression(flat_support[:1], flat_results, self.kern, noise_var=0.0)

            flat_result = m.posterior_samples_f(flat_support,size=1)[:,:,0]

            self.regression = GPy.models.GPRegression(flat_support, flat_result, self.kern, noise_var=0.0)

    def query(self, queries) -> Tuple[NDArray[Shape["query_nr, ... query_dim"], Number], NDArray[Shape["query_nr, ... result_dim"], Number]]:

        flat_queries = queries.reshape((queries.shape[0], -1))
        
        flat_results, pred_cov = self.regression.predict_noiseless(flat_queries)
        results = flat_results.reshape((queries.shape[0], *self.result_shape))

        return queries, results

    def query_constrain(self) -> QueryConstrain:
        x_min_max = zip(self.min_support, self.max_support)
        query_ranges = np.asarray(tuple((x_min, x_max) for x_min, x_max in x_min_max))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)


    def __call__(self, **kwargs) -> Self:
        obj: GaussianProcessDataSource = super().__call__( **kwargs)
        obj.regression = self.regression
        return obj
    

@dataclass
class BrownianProcessDataSource(GaussianProcessDataSource):
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    min_support: Tuple[float,...] = init(default=(0,))
    max_support: Tuple[float,...] = init(default=(100,))
    brown_var: float = init(default=0.01)

    def post_init(self):
        self.kern = GPy.kern.Brownian(variance=self.brown_var)
        super().post_init()
    
@dataclass
class BrownianDriftDataSource(GaussianProcessDataSource):
    query_shape: Tuple[int,...] = init(default=(2,))
    result_shape: Tuple[int,...] = init(default=(1,))
    brown_var: float = init(default=0.01) #0.005
    rbf_var: float = init(default=0.25)
    rbf_leng: float = init(default=0.1) #0.4
    min_support: Tuple[float,...] = init(default=(0,-1))
    max_support: Tuple[float,...] = init(default=(2000,1))

    def post_init(self):
        self.kern = GPy.kern.Brownian(active_dims=[0],variance=self.brown_var)*GPy.kern.RBF(input_dim=1, lengthscale=self.rbf_leng, variance=self.rbf_var, active_dims=[1])+GPy.kern.RBF(input_dim=1, lengthscale=self.rbf_leng, variance=self.rbf_var, active_dims=[1])
        super().post_init()

@dataclass
class RBFDriftDataSource(GaussianProcessDataSource):
    query_shape: Tuple[int,...] = init(default=(2,))
    result_shape: Tuple[int,...] = init(default=(1,))
    brown_var: float = init(default=0.01) #0.005
    rbf_var: float = init(default=0.25)
    rbf_leng: float = init(default=0.1) #0.4
    min_support: Tuple[float,...] = init(default=(0,-1))
    max_support: Tuple[float,...] = init(default=(2000,1))

    def post_init(self):
        self.kern = GPy.kern.RBF(input_dim=1, active_dims=[0],variance=self.rbf_var*2, lengthscale=self.brown_var*2000)*GPy.kern.RBF(input_dim=1, lengthscale=self.rbf_leng, variance=self.rbf_var, active_dims=[1])+GPy.kern.RBF(input_dim=1, lengthscale=self.rbf_leng, variance=self.rbf_var, active_dims=[1])
        super().post_init()

@dataclass
class SinDriftDataSource(GaussianProcessDataSource):
    query_shape: Tuple[int,...] = init(default=(2,))
    result_shape: Tuple[int,...] = init(default=(1,))
    brown_var: float = init(default=0.005) #0.005
    rbf_var: float = init(default=0.25)
    rbf_leng: float = init(default=0.1) #0.4
    min_support: Tuple[float,...] = init(default=(0,-1))
    max_support: Tuple[float,...] = init(default=(2000,1))


    def post_init(self):
        self.kern = GPy.kern.Cosine(input_dim=1, active_dims=[0],variance=self.rbf_var*2, lengthscale=self.brown_var*2000)*GPy.kern.RBF(input_dim=1, lengthscale=self.rbf_leng, variance=self.rbf_var, active_dims=[1])+GPy.kern.RBF(input_dim=1, lengthscale=self.rbf_leng, variance=self.rbf_var, active_dims=[1])
        super().post_init()

@dataclass
class MixedDriftDataSource(GaussianProcessDataSource):
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
        x_min_max = zip(self.min_support, self.max_support)
        query_ranges = np.asarray(tuple((x_min, x_max) for x_min, x_max in x_min_max))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)

@dataclass
class MixedBrownDriftDataSource(GaussianProcessDataSource):
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
        x_min_max = zip(self.min_support, self.max_support)
        query_ranges = np.asarray(tuple((x_min, x_max) for x_min, x_max in x_min_max))
        return QueryConstrain(count=None, shape=self.query_shape, ranges=query_ranges)

@dataclass
class TimeBehaviorDataSource(TimeDataSource):

    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    behavior: DataBehavior = init()
    change_times: NDArray[Shape["change_times"], Number] = post_init()
    change_values: NDArray[Shape["change_values"], Number] = post_init()
    current_time: float = pre_init(default=0)

    def post_init(self):
        super().post_init()
        self.behavior = is_set(self.behavior)()
        self.change_times, self.change_values = self.behavior.behavior()


    @property
    def exhausted(self):
        return self.current_time < self.behavior.stop_time

    def query(self, queries: NDArray[ Shape["query_nr, ... query_dim"], Number]) -> Tuple[NDArray[Shape["query_nr, ... query_dim"], Number], NDArray[Shape["query_nr, ... result_dim"], Number]]:
        times = queries
        self.current_time = times[-1,0]

        indices = np.searchsorted(self.change_times, times[...,0],side='right') -1

        results = self.change_values[indices][:,None]

        return queries, results