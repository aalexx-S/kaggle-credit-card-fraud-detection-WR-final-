import numpy as np
from sklearn import linear_model

def train(X, y, class_weight, n_iter):
    clf = linear_model.SGDClassifier(class_weight=class_weight, n_iter=n_iter)
    clf.fit(X, y)
    return clf
