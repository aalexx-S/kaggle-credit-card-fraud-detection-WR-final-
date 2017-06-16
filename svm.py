from sklearn import svm

def train(X, y, mode):
    clf = svm.SVC()
    if mode == 'SVC':
        clf = svm.SVC()
    clf.fit(X, y)
    return clf
