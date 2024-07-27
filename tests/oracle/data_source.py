from __future__ import annotations
from typing import TYPE_CHECKING

from alts.core.oracle.data_source import DataSource
import alts.modules.oracle.data_source as dsm
from alts.core.data.constrains import QueryConstrain, ResultConstrain
from alts.core.configuration import pre_init, is_set, init, post_init, Configurable

import numpy as np
import GPy
import pytest

if TYPE_CHECKING:
    from typing import Tuple, List, Any, Type
    from nptyping import  NDArray, Number, Shape

"""
Structure: 
One Dimensional Test
Multidimensional Test

"""
class DataSourceTest(Configurable):
    ds : DataSource = init(default_factory=dsm.RandomUniformDataSource)

    def test(self, values : NDArray[Shape["query_dim, result_dim"], Number]): # type: ignore
        pass


if __name__ == '__main__':
    dst = DataSourceTest()()
    print(dst.ds.default_factory.__annotations__)
    print(dsm.RandomUniformDataSource.u)
