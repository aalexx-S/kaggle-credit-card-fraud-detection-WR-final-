import numpy as np
from sklearn import linear_model

def train(X, y):
    clf = linear_model.SGDClassifier()
    clf.fit(X, y)
    return clf
