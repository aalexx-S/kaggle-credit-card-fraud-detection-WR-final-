from .baseSampler import BaseSampler
import random
import numpy as np
from collections import defaultdict

class Naive(BaseSampler):
    """
    An oversampler using naive method. Only oversample the class with fewest
    data.
    """
    config = None

    def __init__(self, config):
        self.config = self._Parameters(config)

    def fit_sample(self, X, y):
        # decide which class to oversampler
        label_d = defaultdict(lambda: 0)
        for i in y:
            label_d[i] += 1
        label = None
        max_c = float('inf')
        for i, j in label_d.items():
            if max_c > j:
                label = i
                max_c = j
        # sampling
        pool = [X[i] for i in range(len(y)) if y[i] == label]
        size = int(len(pool) * (self.config.ratio-1))
        tmp = [random.choice(pool) for i in range(size)]
        return np.concatenate((X.copy(), tmp)),\
                    np.concatenate((y.copy(), [label]*size))


    class _Parameters:
        ratio=None

        def __init__(self, config):
            self.ratio = float(config.get('SAMPLER', 'ratio'))
