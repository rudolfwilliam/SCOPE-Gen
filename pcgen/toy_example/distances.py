from pcgen.distances import Distance
import numpy as np


class L2Distance(Distance):
    def __call__(self, x, y):
        return np.linalg.norm(x - y, axis=-1)
    