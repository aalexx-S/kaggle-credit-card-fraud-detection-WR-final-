import numpy as np

from .baseSampler import BaseSampler
from pySMOTE import SMOTE


class PySmote(BaseSampler):
    """ An oversampler using smote implemented by us.
    """
    def __init__(self, config):
        self.config = self._Parameters(config)

    def fit_sample(self, X, y):
        smote = SMOTE(ratio=(self.config.ratio-1)*100,
                      k_neighbors=self.config.k_neighbors)

        # extract minor samples
        minor_samples = []
        major_samples = []
        for sample, label in zip(X, y):
            if label == 1:
                minor_samples.append(sample)
            elif label == 0:
                major_samples.append(sample)

        minor_sample_count = len(minor_samples)

        new_minor_samples = smote.oversample(minor_samples, merge=True)
        count_diff = new_minor_samples.shape[0] - minor_sample_count

        for i in range(count_diff):
            y.append(1)

        return np.concatenate((major_samples, new_minor_samples)), y


    class _Parameters:
        ratio=None

        def __init__(self, config):
            self.ratio = float(config.get('SAMPLER', 'ratio'))
            self.k_neighbors = int(config.get('SAMPLER', 'k_neighbors'))
