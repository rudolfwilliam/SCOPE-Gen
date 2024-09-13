"""Simple farthest point sampling algorithm. Required for distance pruning."""

from random import choice
import numpy as np


def fps_step(left_over_set, sub_set, distance_func):
    """
    Perform one step of farthest point sampling.

    left_over_set: set of points to sample from
    sub_set: set of points already sampled
    distance_func: function to compute distance between points
    """
    if len(sub_set) == 0:
        idx = choice(range(len(left_over_set)))
        return left_over_set[idx], idx
    if len(left_over_set) == 0:
        raise ValueError("No points left to sample from.")
    farthest_point = None
    max_distance = 0
    for idx, point in enumerate(left_over_set):
        min_distance = min([distance_func(point, sub_point) for sub_point in sub_set])
        if min_distance >= max_distance:
            max_distance = min_distance
            farthest_point = point
            farthest_idx = idx
    return farthest_point, farthest_idx
