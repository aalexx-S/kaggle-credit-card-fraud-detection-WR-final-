from .baseClassifier import BaseClassifier
from sklearn.ensemble import RandomForestClassifier
from ast import literal_eval

class RandomForest(BaseClassifier):
    config = None
    clf = None

    def __init__(self, config):
        self.config = self._Parameters(config)
        self.clf = \
                RandomForestClassifier(**self.config.kwargs)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
    

    class _Parameters:
        con_sec = 'RANDOM FOREST'
        kwargs = {}

        def __init__(self, config):
            n_estimator = config.get(self.con_sec, 'n_estimators')
            if n_estimator != '':
                self.kwargs['n_estimators'] = int(n_estimator)
            max_depth = config.get(self.con_sec, 'max_depth')
            if max_depth != '':
                self.kwargs['max_depth'] = int(max_depth)
            class_weight = config.get(self.con_sec, 'class_weight')
            if class_weight != '':
                self.kwargs['class_weight'] = literal_eval(class_weight)
