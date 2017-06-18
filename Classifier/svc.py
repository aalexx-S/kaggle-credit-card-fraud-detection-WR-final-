from .baseClassifier import BaseClassifier
from sklearn import svm

class SVC(BaseClassifier):
    config=None
    clf=None

    def __init__(self, config):
        self.config = self._Parameters(config)

    def fit(self, X, y):
        self.clf = svm.SVC()
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)


    class _Parameters:

        def __init__(self, config):
            pass

