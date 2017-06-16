from imblearn.over_sampling import SMOTE
import random

def over_sampling_SMOTE_imblearn(X, y, kind):
    sm = SMOTE(kind=kind)
    return sm.fit_sample(X, y)

def over_sampling_naive(X, y, ratio):
    pool = [X[i] for i in range(len(y)) if y[i] == 1]
    size = len(pool)
    tmp = []
    for i in range(size):
        tmp.append(pool[random.randint(0, len(pool)-1)])
    return X.extend(tmp), y.extend([1]*size)
