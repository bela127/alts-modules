#Fully documented as of 20.07.2024
"""
:doc:`Core Module </core/oracle/augmentation>`
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass
from alts.core.oracle.augmentation import Augmentation
from alts.core.configuration import init


import numpy as np

if TYPE_CHECKING:
    from typing_extensions import Self
    from typing import Tuple
    from nptyping import NDArray, Number, Shape

@dataclass
class NoiseAugmentation(Augmentation):
    """
    | **Description**
    |   Adds noise to the results of the augmented :doc:`Data Source </core/oracle/data_source>`.

    :param noise_ratio: Standard deviation from actual result
    :type noise_ratio: ``float``
    """
    noise_ratio: float = init(default=0.01) 

    rng = np.random.default_rng()

    def query(self, queries: NDArray[ Shape["query_nr, ... query_dim"], Number]) -> Tuple[NDArray[Shape["query_nr, ... query_dim"], Number], NDArray[Shape["query_nr, ... result_dim"], Number]]: # type: ignore
        """
        | **Description**
        |   Applies random noise with the given standard deviation ``noise_ratio`` to the result.

        :param queries: Requested Query
        :type queries: `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
        :return: Processed Query, Result 
        :rtype: A tuple of two `NDArray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_  
        """
        queries, results = self.data_source.query(queries)
        augmented = self.rng.normal(results, self.noise_ratio) # type: ignore
        return queries, augmented