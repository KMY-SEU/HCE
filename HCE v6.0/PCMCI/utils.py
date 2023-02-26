import numpy as np


def recall(true, pred):
    TP = 0  # true positive
    FN = 0  # false negative

    #
    coor = np.argwhere(true == 1)
    for c in coor:
        if pred[c[0], c[1], c[2]] == 1:
            TP += 1
        else:
            FN += 1

    return TP / (TP + FN)


def false_positive_rate(true, pred):
    #
    FP = 0  # false positive
    TN = 0  # true negative

    #
    cs = np.argwhere(true == 0)
    for c in cs:
        if pred[c[0], c[1], c[2]] == 1:
            FP += 1
        else:
            TN += 1

    return FP / (FP + TN)
