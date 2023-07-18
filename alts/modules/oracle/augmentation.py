from dataclasses import dataclass
from alts.core.oracle.augmentation import Augmentation

import numpy as np

@dataclass
class NoiseAugmentation(Augmentation):
    noise_ratio: float = 0.01

    rng = np.random.default_rng()

    def query(self, queries: NDArray[ Shape["query_nr, ... query_dim"], Number]) -> Tuple[NDArray[Shape["query_nr, ... query_dim"], Number], NDArray[Shape["query_nr, ... result_dim"], Number]]:
        queries, results = self.data_source.query(queries)
        augmented = self.rng.normal(results, self.noise_ratio)
        return queries, augmented