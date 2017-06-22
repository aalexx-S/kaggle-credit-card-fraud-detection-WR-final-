from .baseSampler import BaseSampler
from imblearn.over_sampling import ADASYN

class ADASYNImlearn(BaseSampler):
    """
    An oversampler using ADASYN implemented by imlearn.
    """
    def __init__(self, config):
        self.config = self._Parameters(config)

    def fit_sample(self, X, y):
        ada = ADASYN()
        return ada.fit_sample(X, y)

    class _Parameters:
        kind = None

        def __init__(self, config):
            self.kind = config.get('SAMPLER', 'kind')
