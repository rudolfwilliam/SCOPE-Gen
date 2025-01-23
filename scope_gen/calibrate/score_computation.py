"""Compute non-conformity scores for data points."""

import numpy as np
from scope_gen.utils import slice_dict
from scope_gen.scripts.remove_dupl import remove_dupl


class ScoreComputer(object):
    """Compute scores for calibration."""
    def __init__(self, prediction_pipeline, order_func, score_func, adjust_for_dupl=False):
        self.order_func = order_func
        self.prediction_pipeline = prediction_pipeline
        self.score_func = score_func
        self.adjust_for_dupl = adjust_for_dupl

    def __call__(self, data):
        return self.compute_scores(data)
    
    def compute_scores(self, idxs, return_idxs=False):
        scores = []
        indices = []
        for data_idx in idxs:
            if return_idxs:
                score, first_idx = self._compute_score(data_idx, return_idx=True)
                indices.append(first_idx)
            else:
                score = self._compute_score(data_idx)
            scores.append(score)

        scores_array = np.array(scores)
        if return_idxs:
            indices_array = np.array(indices)
            return scores_array, indices_array
        return scores_array

    def _compute_score(self, data_idx, return_idx=False):
        instance = self.prediction_pipeline.generate(data_idx)
        if instance is None:
            # no instance could be generated (can only happen in filter stage)
            if return_idx:
                return np.nan, 0
            return np.nan
        if not any(instance["labels"] == 1):
            if return_idx:
                # all instances were checked
                if self.order_func is None:
                    return np.inf, get_first_idx(instance, admissible=False, adjust_for_dupl=self.adjust_for_dupl)
                return np.nan, get_first_idx(instance, admissible=False, adjust_for_dupl=self.adjust_for_dupl)
            else:
                if self.order_func is None:
                    return np.inf
                return np.nan
        if self.order_func is not None:
            instance_ordered = self.order_func(instance)
        else:
            instance_ordered = instance
        # return up to first admission, counting duplicates
        first_idx = np.where(instance_ordered["labels"] == 1)[0][0]
        # compute score
        score = self.score_func(slice_dict(instance_ordered, first_idx))
        if return_idx:
            # get_first_idx can adjust for duplicates!
            return score, get_first_idx(instance, admissible=True, adjust_for_dupl=self.adjust_for_dupl)
        return score


def get_first_idx(instance, admissible=True, adjust_for_dupl=False):
    """Get the index of the first admission."""
    if not admissible:
        if adjust_for_dupl:
            return len(remove_dupl(instance)["labels"])
        else:
            return len(instance["labels"])
    else:
        if adjust_for_dupl:
            # there seems to be a bug in the original code here 
            # - if two answers are the same, they should have the same label
            # - this is a conservative workaround
            new_instance = remove_dupl(instance)
            if not(any(new_instance["labels"] == 1)):
                return np.where(instance["labels"] == 1)[0][0] + 1
            return np.where(new_instance["labels"] == 1)[0][0] + 1
        else:
            return np.where(instance["labels"] == 1)[0][0] + 1
        