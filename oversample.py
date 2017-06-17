from imblearn.over_sampling import SMOTE
import random

def over_sampling_SMOTE_imblearn(X, y, kind):
    sm = SMOTE(kind=kind)
    return sm.fit_sample(X, y)

def over_sampling_naive(X, y, ratio):
    pool = [X[i] for i in range(len(y)) if y[i] == 1]
    size = int(len(pool) * (ratio-1))
    tmp = [random.choice(pool) for i in range(size)]
    reX = X.copy()
    rey = y.copy()
    reX.extend(tmp)
    rey.extend([1]*size)
    return reX, rey
