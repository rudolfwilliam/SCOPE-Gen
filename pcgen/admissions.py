"""Admission functions."""


class Admission(object):
    """Base class for admission functions."""
    @property
    def gt(self):
        if not hasattr(self, '_gt'):
            import warnings
            warnings.warn("Ground truth not set. Setting to None.")
            self._gt = None
        return self._gt
    
    @gt.setter
    def gt(self, value):
        self._gt = value
    
    def __call__(self, x):
        """Return True if the prediction is admissible, False otherwise."""
        raise NotImplementedError


class ProximalAdmission(Admission):
    """Measures whether a prediction is within a certain distance of the ground."""
    def __init__(self, epsilon, distance_func):
        self.epsilon = epsilon
        self.distance_func = distance_func
        super(ProximalAdmission, self).__init__()

    def __call__(self, x):
        return self.distance_func(x, self.gt) <= self.epsilon
    