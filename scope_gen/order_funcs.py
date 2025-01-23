"""This module contains functions to order the instances in the dataset according to the used filter."""

import numpy as np
from scope_gen.utils import order_dict


def quality_order_func(instance):
    """Order the instance according to the quality of the instances."""
    scores = instance["scores"]
    idxs = np.argsort(-scores, kind="stable")
    instance_ordered = order_dict(instance, idxs)
    return instance_ordered


def distance_order_func(instance):
    """Order the instance according to the distance of the instances."""
    similarities = instance["similarities"]
    n = len(instance["labels"])
    if n == 0:
        return instance
    # Initialize the list of selected points' indices
    idxs = []
    
    # Randomly select the first point and add its index to the list
    idx = np.random.randint(n)
    idxs.append(idx)

    # Initialize a list to store the maximum distance to the selected points for each point
    max_distances = np.full(n, np.inf)
    max_distances[idx] = -np.inf

    # Iteratively select the next point
    for _ in range(1, n):
        last_index = idxs[-1]
        
        # Update the maximum distance to the selected set for each point
        for i in range(n):
            if i not in idxs:
                dist = -similarities[last_index, i]
                max_distances[i] = min(max_distances[i], dist)
        
        # Select the point with the maximum distance to the current set of selected points
        next_index = np.argmax(max_distances)
        max_distances[next_index] = -np.inf
        idxs.append(next_index)
    
    instance_ordered = order_dict(instance, idxs)
    return instance_ordered
