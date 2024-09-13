"""Basic calibrations functions."""

import numpy as np


def calibrate(data, score):
    scores = []
    for i in range(1, len(data)):
        x, cond = data[i]
        # Assuming `x` and `cond` are NumPy arrays.
        score_ = score(x[np.newaxis, :], cond[np.newaxis, :])
        scores.append(score_)
    return np.stack(scores)


def get_percentile(scores, alpha, max_value=1e10):
    if all(np.isnan(scores)) or all(np.isinf(scores)):
        return np.nan
    # remove nan values
    scores = scores[~np.isnan(scores)]
    n = scores.shape[0]
    perc = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    # unfortunately, np.quantile does not handle inf automatically
    scores = np.where(np.isinf(scores), max_value, scores)
    quant = np.quantile(scores, perc, method='higher')
    # if quantile is the max value, return nan
    if quant == max_value:
        return np.nan
    else:
        return quant
