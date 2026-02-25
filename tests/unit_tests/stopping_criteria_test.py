#Test version 1.1 as of 22.06.2025
from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from alts.core.configuration import pre_init, is_set, init, post_init

from alts.core.stopping_criteria import StoppingCriteria
import alts.modules.stopping_criteria as scs

from alts.core import experiment
from alts.modules import blueprint
from alts.modules.oracle.data_source import RandomUniformDataSource
from alts.core.oracle.data_source import DataSource
from alts.core.data_process.time_source import TimeSource
import alts.modules.data_process.time_source as ts
from alts.modules.data_process.process import DataSourceProcess

from dataclasses import dataclass
if TYPE_CHECKING:
    from typing import Tuple

import alts.modules.testing_modules as tm
import numpy as np
import pytest

"""
| **Test aims**
|   The stopping criterias modules are tested for:
|   - correct triggering of the stopping criteria
"""

@dataclass
class ExhaustedDataSource(RandomUniformDataSource):
    """
    ExhaustedDataSource(query_shape, result_shape, a, b, exhausted_time)
    | **Description**
    |   A LineDataSource that exhausts after a given amount of queries.

    :param query_shape: The expected shape of the queries (default= (1,))
    :type query_shape: tuple of ints
    :param result_shape: The expected shape of the results (default= (1,))
    :type result_shape: tuple of ints
    :param u: Max result value, (default= 1)
    :type u: float (optional)
    :param l: Min result value, (default= 0)
    :type l: float (optional)
    :param exhaust_in: How many queries until DataSource is exhausted (default= 5)
    """
    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    a: float = init(default=1)
    b: float = init(default=0)
    exhaust_in: float = init(default=5)

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
        self.exhaust_in -= 1
        results = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*self.a) + np.ones(self.result_shape)*self.b # type: ignore
        return queries, results
    
    @property
    def exhausted(self) -> bool:
        """
        exhausted(self) -> bool
        | **Description**
        |   A ``DataSource`` is exhausted if all its available data has been querried.

        :return: Whether the ``DataSource`` has been exhausted
        :rtype: ``boolean``
        """
        return True if self.exhaust_in <= 0 else False
    

stopping_criterias: list[type[StoppingCriteria]] = [
    scs.TimeStoppingCriteria,
    scs.DataExhaustedStoppingCriteria
]

@pytest.mark.parametrize("sc", stopping_criterias)
def test_exhausted(sc: type[StoppingCriteria]):
    """
    | **Description**
    |   Tests for correct triggering of the stopping criteria
    """
    if sc == scs.TimeStoppingCriteria:
        #Should trigger after given time
        #Normal time 1
        bp = tm.TestBlueprint(stopping_criteria=sc(stop_time=100))
        exp = experiment.Experiment(bp, 1)
        exp.run()
        assert exp.time_source.time == 101
        #Normal time 2
        bp = tm.TestBlueprint(stopping_criteria=sc(stop_time=220))
        exp = experiment.Experiment(bp, 1)
        exp.run()
        assert exp.time_source.time == 221
        #Zero time
        bp = tm.TestBlueprint(stopping_criteria=sc(stop_time=0))
        exp = experiment.Experiment(bp, 1)
        exp.run()
        assert exp.time_source.time == 1
    elif sc == scs.DataExhaustedStoppingCriteria:
        #Should trigger when DataSource is exhausted
        #Normal case 1
        bp = tm.TestBlueprint(stopping_criteria=sc(), process=DataSourceProcess(ExhaustedDataSource(exhaust_in=10)))
        exp = experiment.Experiment(bp, 1)
        exp.run()
        assert exp.process.data_source.exhaust_in == 0 # type: ignore
        #Normal case 2
        bp = tm.TestBlueprint(stopping_criteria=sc(), process=DataSourceProcess(ExhaustedDataSource(exhaust_in=3)))
        exp = experiment.Experiment(bp, 1)
        exp.run()
        assert exp.process.data_source.exhaust_in == 0 # type: ignore
        #Edge Case 1: exhausts in one query
        bp = tm.TestBlueprint(stopping_criteria=sc(), process=DataSourceProcess(ExhaustedDataSource(exhaust_in=1)))
        exp = experiment.Experiment(bp, 1)
        exp.run()
        assert exp.process.data_source.exhaust_in == 0 # type: ignore
    else:
        raise ValueError(f"Stopping Criteria not found: {sc}")
