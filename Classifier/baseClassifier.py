class BaseClassifier:
    def fit(self, X, y):
        pass

    def predict(self, X):
        return [0]*len(X)

    def setArgs(self, **kwargs):
        return
