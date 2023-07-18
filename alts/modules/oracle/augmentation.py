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
    noise_ratio: float = init(default=0.01) 

    rng = np.random.default_rng()

    def query(self, queries: NDArray[ Shape["query_nr, ... query_dim"], Number]) -> Tuple[NDArray[Shape["query_nr, ... query_dim"], Number], NDArray[Shape["query_nr, ... result_dim"], Number]]:
        queries, results = self.data_source.query(queries)
        augmented = self.rng.normal(results, self.noise_ratio)
        return queries, augmented