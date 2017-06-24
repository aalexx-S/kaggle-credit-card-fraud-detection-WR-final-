from .baseClassifier import BaseClassifier
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from ast import literal_eval

class BaggingSVC(BaseClassifier):
    config=None
    clf=None

    def __init__(self, config):
        self.config = self._Parameters(config)

    def fit(self, X, y):
        n_estimators = 10
        self.clf = BaggingClassifier(base_estimator=SVC(),
                                     max_samples=1.0/n_estimators,
                                     n_estimators=n_estimators,
                                     n_jobs=-1)
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)


    class _Parameters:

        def __init__(self, config):
            pass
