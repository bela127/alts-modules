from __future__ import annotations
from typing import TYPE_CHECKING

from alts.core.oracle.data_source import DataSource
import alts.modules.oracle.data_source as dsm
from alts.core.data.constrains import QueryConstrain, ResultConstrain
from alts.core.configuration import pre_init, is_set, init, post_init, Configurable

import numpy as np
import GPy

if TYPE_CHECKING:
    from typing import Tuple, List, Any, Type
    from nptyping import  NDArray, Number, Shape

"""
Structure: 
General loop
General Parameters
Specific Parameters
Specific Value Ranges
General Edgecases
Specific Edgecases
"""
class DataSourceTest(Configurable):
    ds : DataSource = init(default_factory=dsm.RandomUniformDataSource)

    def assign_module(self, pmodule : DataSource):
        self.ds = pmodule

    def test(self, loops : int = 10000) -> bool:
        data : list = []
        result : float
        rand = ds.RandomUniformDataSource(query_shape = self.ds.query_shape, result_shape = self.ds.result_shape,l = self.ds.query_constrain().ranges[0][0] , u = self.ds.query_constrain().ranges[0][1])()
        for i in range(loops):
            a = rand.query(np.asarray((1,)) * float(i)/loops)[0]
            try:
                result = self.ds.query(a)[0]
                if result < self.ds.result_constrain().ranges[0][0] or result >= self.ds.result_constrain().ranges[0][1]:
                    raise ValueError
                data.append(result)
            except ValueError:
                print("ValueError on query: {} \nWith the result: {} \nWith the settings: {}, {}".format(a, result, self.ds.query_constrain(), self.ds.result_constrain()))
                return False
            except Exception as ex:
                print("{} on query: {} \n With the settings: {}, {}".format(ex, a, self.ds.query_constrain(), self.ds.result_constrain()))
                return False
        print()
        return True

if __name__ == '__main__':
    dst = DataSourceTest()()
    print(dst.ds.default_factory.__annotations__)
