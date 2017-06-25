from .baseSampler import BaseSampler
from imblearn.under_sampling import NearMiss

class NearMissImlearn(BaseSampler):
    """
    An undersampler using NearMiss implemented by imlearn.
    """
    def __init__(self, config):
        self.config = self._Parameters(config)

    def fit_sample(self, X, y):
        nm = NearMiss(version=3, n_neighbors=7, n_neighbors_ver3 = 5)
        return nm.fit_sample(X, y)

    class _Parameters:
        kind = None

        def __init__(self, config):
            self.kind = config.get('SAMPLER', 'kind')
