from numpy.random import choice
from numpy import array
from numpy import concatenate
from collections import defaultdict

from .baseSampler import BaseSampler

class NaiveUS(BaseSampler):
    """NaiveUS: naive under sampler
    only undersample the majority class
    """
    def __init__(self, config):
        self.ratio = int(config.get('SAMPLER','ratio'))

    def fit_sample(self, X, y):
        # choose label
        tmpd = defaultdict(lambda: 0)
        for i in y:
            tmpd[i] += 1
        label = None
        maxt = -float('inf')
        for i, j in tmpd.items():
            if j > maxt:
                maxt = j
                label = i
        # sample
        pool = [i for i, j in zip(X, y) if j == label]
        otherX, othery = zip(*[(i, j) for i, j in zip(X, y) if j != label])
        size = int(len(pool) / self.ratio)
        if size == 0:
            size = 1
        index = choice(range(len(pool)), size, False)
        return concatenate((array(pool)[index], list(otherX))),\
                            concatenate(([label]*size, list(othery)))
