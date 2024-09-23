"""Non-conformity scores."""

from pcgen.utils import slice_dict
from typing import Any
import numpy as np

SCORES = ["count", "sum", "max", "min", "distance"]

class NonConformityScore(object):
    def __init__(self):
        super(NonConformityScore, self).__init__()
    
    def reset(self):
        raise NotImplementedError
    
    def update(self, x):
        return self.compute_score()
    
    def compute_score(self):
        raise NotImplementedError


class CountScore(NonConformityScore):
    def __init__(self):
        super(CountScore, self).__init__()
    
    def __call__(self, instance, apply_seq=False):
        if apply_seq:
            return np.arange(1, len(instance["scores"]) + 1)
            #return np.arange(0, len(instance["scores"]))
        else:
            return len(instance["scores"])


class SumScore(object):
    def __init__(self, gamma=0.5):
        self.gamma = gamma
        super(SumScore, self).__init__()
    
    def __call__(self, instance, apply_seq=False):
        if apply_seq:
            return np.array([sum(instance["scores"][:(i + 1)]) + self.gamma * len(instance["scores"][:(i + 1)]) for i in range(0, len(instance["scores"]))])
        else:
            return sum(instance["scores"]) + self.gamma * len(instance["scores"])


class MaxScore(object):
    def __init__(self, gamma=0.1):
        self.gamma = gamma
        super(MaxScore, self).__init__()
    
    def __call__(self, instance, apply_seq=False):
        if apply_seq:
            return np.array([max(instance["scores"][:(i + 1)]) + self.gamma * len(instance["scores"][:(i + 1)]) for i in range(0, len(instance["scores"]))])
        else:
            return max(instance["scores"]) + self.gamma * len(instance["scores"])
        
class MinScore(object):
    def __init__(self):
        super(MinScore, self).__init__()
    
    def __call__(self, instance, apply_seq=False):
        if apply_seq:
            return np.array([-min(instance["scores"][:(i + 1)]) for i in range(0, len(instance["scores"]))])
        else:
            return -min(instance["scores"])

class DistanceScore(object):
    def __init__(self):
        super(DistanceScore, self).__init__()
    
    def __call__(self, instance, apply_seq=False):
        if apply_seq:
            return np.array([self(slice_dict(instance, i)) for i in range(0, len(instance["similarities"]))])
        else:
            if len(instance["similarities"]) > 1:
                # fill diagonal with 0
                temp = np.copy(instance["similarities"])
                np.fill_diagonal(temp, 0)
                return np.max(temp)
            else:
                return -1.
        