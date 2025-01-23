import numpy as np


class Distance(object):
    def __call__(self, x, y):
        raise NotImplementedError
    
    def distance_matrix(self, xs):
        n = len(xs)
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = self(xs[i], xs[j])
                dists[i, j] = dist
                dists[j, i] = dist
        return dists
    