from sklearn.linear_model import LogisticRegression as SKLR

from .baseClassifier import BaseClassifier

class LogisticRegression(BaseClassifier):

    def __init__(self, config):
        self.config = self._Parameters(config)

    def fit(self, X, y):
        self.clf = SKLR(**self.config.kwargs)
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)


    class _Parameters:

        def __init__(self, config):
            self.kwargs={}

