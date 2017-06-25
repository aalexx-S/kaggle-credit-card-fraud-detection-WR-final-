from .baseSampler import BaseSampler
from imblearn.under_sampling import NearMiss

class NearMissImlearn(BaseSampler):
    """
    An undersampler using NearMiss implemented by imlearn.
    """
    def __init__(self, config):
        self.config = self._Parameters(config)

    def fit_sample(self, X, y):
        nm = NearMiss(version=3)
        return nm.fit_sample(X, y)

    class _Parameters:
        kind = None

        def __init__(self, config):
            self.kind = config.get('SAMPLER', 'kind')
