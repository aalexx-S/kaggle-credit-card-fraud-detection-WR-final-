from .baseClassifier import BaseClassifier
import numpy as np
from sklearn import linear_model
from ast import literal_eval

class SGD(BaseClassifier):

    def __init__(self, config):
        self.config = self._Parameters(config)

    def fit(self, X, y):
        self.clf = linear_model.SGDClassifier(**self.config.kwargs)
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)


    class _Parameters:

        def __init__(self, config):
            self.kwargs = {}
            class_weight = config.get('SGD', 'class_weight')
            if class_weight not in ['balanced', '']:
                class_weight = literal_eval(class_weight)
            if class_weight != '':
                self.kwargs['class_weight'] = class_weight
            self.kwargs['n_iter'] = int(config.get('SGD', 'n_iter'))
