from .baseSampler import BaseSampler
from imblearn.under_sampling import RandomUnderSampler

class RUSImlearn(BaseSampler):
    """
    An undersampler using RUS implemented by imlearn.
    """
    def __init__(self, config):
        self.config = self._Parameters(config)

    def fit_sample(self, X, y):
        rus = RandomUnderSampler()
        return rus.fit_sample(X, y)

    class _Parameters:
        kind = None

        def __init__(self, config):
            self.kind = config.get('SAMPLER', 'kind')
