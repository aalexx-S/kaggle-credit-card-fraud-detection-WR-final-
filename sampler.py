import Sampler
from ast import literal_eval

class Smp:
    config = None
    sampler = None

    def __init__(self, config):
        self.config = self._Parameters(config)
        if self.config.oversample_method == 'SMOTE_imlearn':
            self.sampler = Sampler.SMOTEImlearn(config)
        elif self.config.oversample_method == 'naive':
            self.sampler = Sampler.Naive(config)
        elif self.config.oversample_method == 'ADASYN':
            self.sampler = Sampler.ADASYNImlearn(config)
        elif self.config.oversample_method == 'pySMOTE':
            self.sampler = Sampler.PySmote(config)

    def fit_sample(self, X, y):
        return self.sampler.fit_sample(X, y)


    class _Parameters:
        oversample_method = ''
        allowed = []

        def __init__(self, config):
            self.oversample_method = config.get('SAMPLER', 'method')
            self.allowed = literal_eval(config.get('SAMPLER', 'allowed'))
            if self.oversample_method not in self.allowed:
                raise ValueError('Oversample method not allowed by'
                    + ' config.SAMPLER.allowed.')
