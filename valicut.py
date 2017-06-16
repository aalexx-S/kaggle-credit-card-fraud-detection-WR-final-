# This file is named by Louis Tsai
import random

def valicut(X, y, vali_ratio):
    size = int(y.count(1) * vali_ratio)
    ind = [i for i in range(len(y)) if y[i] == 1]
    sampled = random.sample(ind, size)
    ind = [i for i in range(len(y)) if y[i] == 0]
    sampled.extend(random.sample(ind, size))
    val_x, val_y = zip(*[(X[i], y[i]) for i in sampled])
    train_X, train_y\
        = zip(*[(X[i], y[i]) for i in range(len(y)) if i not in sampled])
    return list(train_X), list(train_y), list(val_x), list(val_y)
