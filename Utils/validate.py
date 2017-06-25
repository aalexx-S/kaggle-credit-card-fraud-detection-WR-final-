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
    string = ''
    string += 'ans\pred   True     False  \n'
    string += '===========================\n'
    string += 'True  =     {0}      {1}\n'.format(table[11], table[10])
    string += 'False =     {0}      {1}\n'.format(table[1], table[0])

    return string

def f1_score(table):
    TP = table[11]
    FP = table[1]
    FN = table[10]
    TN = table[0]
    if ((TP+FP) == 0):
        Precision = 0
    else:
        Precision = TP / (TP+FP)
    if ((TP+FN) == 0):
        Recall = 0
    else:
        Recall = TP / (TP+FN)
    #special case when Precision and Recall are all zero
    #set F1 score to zero
    if ((Precision+Recall) == 0):
        F1_score = 0
    else:
        F1_score = (2*Precision*Recall) / (Precision+Recall)
    return F1_score, Precision, Recall

def cost_matrix(val_X_amount):
    re = []
    const = 1
    for i in val_X_amount:
         # cost_mat[C_FP,C_FN,C_TP,C_TN]
         re.append([const, i, const, 0])
    return re

def EDCS_cost(ans, pred, val_X_amount):
    cm = cost_matrix(val_X_amount)
    cv = {1:0, 10:1, 11:2, 0:3}
    re = 0
    for i, j, c in zip(ans, pred, cm):
        re += c[cv[10 * i + j]]
    return re
