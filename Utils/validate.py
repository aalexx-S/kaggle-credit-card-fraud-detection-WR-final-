def correction(answer, target):
    cor = 0
    if len(answer) != len(target):
        print("Warning, answer length not equal. In validate.py, correction.")
    for i, j in zip(answer, target):
        if i == j:
            cor += 1
    return float(cor) / len(answer)

def truth_table(answer, target):
    a = {0:0, 10:0, 1:0, 11:0}
    for i, j in zip(answer, target):
        a[10*i + j] += 1
    return a

def print_truth_table(table):
    print('ans\pred   True     False  ')
    print('===========================')
    print('True  =     {0}      {1}'.format(table[11], table[10]))
    print('False =     {0}      {1}'.format(table[1], table[0]))

def f1_score(table):
    TP = table[11]
    FP = table[1]
    FN = table[10]
    TN = table[0]
    Precision = TP / (TP+FP)
    Recall = TP / (TP+FN)
    return ((2*Precision*Recall) / (Precision+Recall)), Precision, Recall
