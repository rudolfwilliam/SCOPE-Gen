"""Class that defines the pipeline for generating prediction set using scope_gen."""

from scope_gen.utils import slice_dict
import numpy as np


class PredictionPipeline(object):
    def __init__(self, prediction_pipeline=None):
        self.prediction_pipeline = prediction_pipeline

    def generate(self, data_idx=None):
        raise NotImplementedError

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if self.prediction_pipeline is not None:
            self.prediction_pipeline.data = data
        self._data = data

    @data.getter
    def data(self):
        if self.prediction_pipeline is not None:
            return self.prediction_pipeline.data
        return self._data            
    
    def generate_new(self, data):
        """Predict new data."""
        self.data = data
        predictions = []
        for i in range(len(data)):
            prediction = self.generate(i)
            predictions.append(prediction)
        return predictions

class MinimalPredictionPipeline(PredictionPipeline):
    """Simply returns the data."""
    def __init__(self, data):
        super().__init__(None)
        self.data = data

    def generate(self, data_idx):
        instance = self.data[data_idx]
        return instance


class GenerationPredictionPipeline(PredictionPipeline):
    def __init__(self, prediction_pipeline, conformal_p_g, score_func):
        self.conformal_p = conformal_p_g
        self.score_func = score_func
        super().__init__(prediction_pipeline)

    def generate(self, data_idx=None):
        """Generate prediction set."""
        instance = self.prediction_pipeline.generate(data_idx)
        # return up until accumulated score is greater than conformal_p
        #if instance is None:
        #    return None
        acc_scores = self.score_func(instance, apply_seq=True)
        if not any(acc_scores >= self.conformal_p):
            return None
        if not any(acc_scores <= self.conformal_p):
            return {key : np.array([]) for key, value in instance.items()}
            #return {'labels' : np.array([]), 'scores' : np.array([]), 'similarities' : np.array([])} # return empty set
        else:
            first_idx = np.where(acc_scores <= self.conformal_p)[0][-1]
        # return instance up to first_idx
        instance_sliced = slice_dict(instance, first_idx)
        return instance_sliced


class FilterPredictionPipeline(PredictionPipeline):
    """We use some kind of decorator design pattern to dynamically add filters. The object owns an instance of the its parent class."""
    def __init__(self, prediction_pipeline, conformal_p, score_func, order_func):
        self.conformal_p = conformal_p
        self.score_func = score_func
        # order_func is used to order the instances according to the filter
        self.order_func = order_func
        super().__init__(prediction_pipeline)

    def generate(self, idx=None):
        """Generate prediction set."""
        # generate initial prediction set
        instance = self.prediction_pipeline.generate(idx)
        if self.conformal_p is np.nan:
            return instance
        if instance is None:
            return None
        instance_ordered = self.order_func(instance)
        acc_scores = self.score_func(instance_ordered, apply_seq=True)
        if not any(acc_scores >= self.conformal_p):
            return instance
        if not any(acc_scores <= self.conformal_p):
            #return {'labels' : np.array([]), 'scores' : np.array([]), 'similarities' : np.array([])} # return empty set
            return {key : np.array([]) for key, value in instance.items()}
        else:
            first_idx = np.where(acc_scores <= self.conformal_p)[0][-1]
        instance_sliced = slice_dict(instance_ordered, first_idx)
        return instance_sliced


class RemoveDuplicatesPredictionPipeline(PredictionPipeline):
    """Remove duplicates from the prediction set."""
    def __init__(self, prediction_pipeline):
        super().__init__(prediction_pipeline)

    def generate(self, idx=None):
        instance = self.prediction_pipeline.generate(idx)
        if instance is None:
            return None
        if instance['labels'].size == 0:
            return instance
        # remove duplicates from similarity matrix
        labels = instance["labels"]
        similarities = instance["similarities"]
        
        # Find indices of unique answers based on similarities
        unique_indices = []
        seen = set()
        
        for i in range(len(labels)):
            if i not in seen:
                unique_indices.append(i)
                # Mark similar answers to be removed from consideration
                for j in range(i + 1, len(labels)):
                    if similarities[i, j] == 1:
                        seen.add(j)
        
        # Create new instance with filtered data
        #new_instance = {
        #    "labels": labels[unique_indices],
        #    "scores": instance["scores"][unique_indices],
        #    "similarities": similarities[np.ix_(unique_indices, unique_indices)]
        #}
        new_instance = {key: value[unique_indices] if len(value.shape) == 1 else \
                        value[np.ix_(unique_indices, unique_indices)] for key, value in instance.items()}
        return new_instance
    