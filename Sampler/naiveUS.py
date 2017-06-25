from numpy.random import choice
from numpy import array

from .baseSampler import BaseSampler

class NaiveUS(BaseSampler):
    """NaiveUS: naive under sampler
    """
    def __init__(self, config):
        self.ratio = int(config.get('SAMPLER','ratio'))

    def fit_sample(self, X, y):
        size = int(len(X) / self.ratio)
        if size == 0:
            size = 1
        index = choice(range(len(X)), size, False)
        return array(X)[index], array(y)[index]
