from __future__ import annotations
from typing import TYPE_CHECKING
from alts.core.data.data_sampler import DataSampler

if TYPE_CHECKING:
    from typing import Tuple, List, Dict
    from nptyping import NDArray, Number, Shape

from alts.core.oracle.interpolation_strategy import InterpolationStrategy

class NoInterpolation(InterpolationStrategy):
    """
    NoInterpolation()
    | **Description**
    |   ``NoInterpolation`` is an interpolator that does nothing to the given data. 

    :param data_sampler: A sample of the data which contains the to-be interpolated data points
    :type data_sampler: :doc:`DataSampler </core/data/data_sampler>`
    """
    ...