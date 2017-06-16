def correction(answer, target):
    cor = 0
    if len(answer) != len(target):
        print("Warning, answer length not equal. In validate.py, correction.")
    for i, j in zip(answer, target):
        if i == j:
            cor += 1
    return float(cor) / len(answer)
