"""Base classes for generative models."""

from pcgen.algorithms.farthest_point_sampling import fps_step
from pcgen.utils import override
import numpy as np


class GenerativeModel(object):
    def sample(self, data_idx):
        raise NotImplementedError


class PruningGenerativeModel(GenerativeModel):
    """Generative model used for pruning steps."""
    def __init__(self, prediction_pipeline, admission_func):
        # this is required to create the superset to sample from
        self.prediction_pipeline = prediction_pipeline
        self.admission_func = admission_func
        self.super_samples = None
    
    def sample(self, num_samples, cond=None):
        if self.super_samples is None:
            raise ValueError("Super sample not set. Please call reset() first.")


class QualityPruningGenerativeModel(PruningGenerativeModel):
    def __init__(self, prediction_pipeline, admission_func, predictor_, k_max=20):
        super().__init__(prediction_pipeline, admission_func)
        self.predictor = predictor_
        self.k_max = k_max

    def sample(self, cond=None):
        super().sample(self.k_max, cond)
        # sample from the super set the element with the highest prediction value and add it to the sample set
        if len(self.super_samples) == 0:
            # if super_samples is empty, return None
            return None
        preds = self.predictor(np.stack(self.super_samples))
        max_value = max(preds)
        max_index = np.where(preds == max_value)[0][0]
        new_sample = self.super_samples.pop(max_index)
        self.samples.append(new_sample)
        return new_sample


class DistancePruningGenerativeModel(PruningGenerativeModel):
    def __init__(self, prediction_pipeline, admission_func, distance_func):
        super().__init__(prediction_pipeline, admission_func)
        self.distance_func = distance_func

    def sample(self, num_samples, cond=None):
        super().sample(num_samples, cond)
        # sample from the super set the element with the largest distance and add it to the sample set
        if len(self.super_samples) == 0:
            # if super_samples is empty, return None
            return None
        new_sample, idx = fps_step(self.super_samples, self.samples, self.distance_func)
        self.super_samples.pop(idx)
        self.samples.append(new_sample)    
        return new_sample


def get_gen_model_cls(filter_str):
    """Return the generative model class from string representation."""
    if filter_str == "quality":
        return QualityPruningGenerativeModel
    elif filter_str == "distance":
        return DistancePruningGenerativeModel
    else:
        raise ValueError(f"Filter string {filter_str} not recognized.")
    