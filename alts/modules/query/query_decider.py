from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field
from abc import abstractmethod

import numpy as np

from alts.core.configuration import Configurable, Required, is_set
from alts.core.query.query_decider import QueryDecider

if TYPE_CHECKING:
    from typing import Tuple, Optional
    from nptyping import NDArray, Number, Shape

@dataclass    
class AllQueryDecider(QueryDecider):
    def decide(self, query_candidates: NDArray[Shape["query_nr, ... query_dims"], Number], scores: NDArray[Shape["query_nr, [query_score]"], Number]) -> Tuple[bool, NDArray[Shape["query_nr, ... query_dims"], Number]]:
        return True, query_candidates

@dataclass
class NoQueryDecider(QueryDecider):
    def decide(self, query_candidates: NDArray[Shape["query_nr, ... query_dims"], Number], scores: NDArray[Shape["query_nr, [query_score]"], Number]) -> Tuple[bool, NDArray[Shape["query_nr, ... query_dims"], Number]]:
        query = np.empty((0, *query_candidates.shape[1:]))
        return False, query

@dataclass
class ThresholdQueryDecider(QueryDecider):
    threshold: float = 0.5

    def decide(self, query_candidates: NDArray[Shape["query_nr, ... query_dims"], Number], scores: NDArray[Shape["query_nr, [query_score]"], Number]) -> Tuple[bool, NDArray[Shape["query_nr, ... query_dims"], Number]]:
        query = query_candidates[scores[:,0]>self.threshold]
        flag = query.shape[0] > 0
        return flag, query
