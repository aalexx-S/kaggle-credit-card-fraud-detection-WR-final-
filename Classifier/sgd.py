from .baseClassifier import BaseClassifier
import numpy as np
from sklearn import linear_model
from ast import literal_eval

class SGD(BaseClassifier):
    config=None
    clf=None

    def __init__(self, config):
        self.config = self._Parameters(config)

    def fit(self, X, y):
        self.clf = linear_model.SGDClassifier(
                class_weight=self.config.class_weight,
                n_iter=self.config.n_iter)
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)


    class _Parameters:
        class_weight=None
        n_iter=None

        def __init__(self, config):
            self.class_weight = config.get('SGD', 'class_weight')
            if self.class_weight != 'balanced':
                self.class_weight = literal_eval(self.class_weight)
            self.n_iter = int(config.get('SGD', 'n_iter'))
