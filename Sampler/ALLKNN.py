from .baseSampler import BaseSampler
from imblearn.under_sampling import AllKNN

class ALLKNNImlearn(BaseSampler):
    """
    An undersampler using ALLKNN implemented by imlearn.
    """
    def __init__(self, config):
        self.config = self._Parameters(config)

    def fit_sample(self, X, y):
        allknn = AllKNN()
        return allknn.fit_sample(X, y)

    class _Parameters:
        kind = None

        def __init__(self, config):
            self.kind = config.get('SAMPLER', 'kind')
