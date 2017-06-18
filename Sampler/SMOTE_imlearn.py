from .baseSampler import BaseSampler
from imblearn.over_sampling import SMOTE

class SMOTEImlearn(BaseSampler):
    """
    An oversampler using smote implemented by imlearn.
    """
    config = None

    def __init__(self, config):
        self.config = self._Parameters(config)

    def fit_sample(self, X, y):
        sm = SMOTE(kind=self.config.kind)
        return sm.fit_sample(X, y)


    class _Parameters:
        kind = None

        def __init__(self, config):
            self.kind = config.get('SAMPLER', 'kind')

