"""Base classes for generative models."""

import numpy as np


class GenerativeModel(object):
    def sample(self, data_idx):
        raise NotImplementedError
    