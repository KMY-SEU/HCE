import numpy as np
import pandas as pd

from sklearn.neighbors import KDTree
from mi_ksg import mi_ksg


def causal_entropy(samples, i, j, K, k_ksg=14):
    """
    compute causal entropy.

    :param samples: the collected samples
    :param i: node i
    :param j: node j
    :param K: conditional set K
    :param k_ksg: the k of knn in KSG method
    :return: causal entropy I(i, j| K)
    """

    # mi of [i, j, K]
    mi_ijK = mi_ksg(samples[:, [i, j] + K], k=k_ksg)

    # mi of [i, K]
    mi_iK = mi_ksg(samples[:, [i] + K], k=k_ksg)

    # mi of [j, K]
    mi_jK = mi_ksg(samples[:, [j] + K], k=k_ksg)

    if len(K) == 1:
        # I(X, Y, Z) = I(X, Y, Z) - I(X, Z) - I(Y, Z)
        # return np.maximum(0, mi_ijK - mi_iK - mi_jK)
        return mi_ijK - mi_iK - mi_jK
    else:
        # I(X, Y, ..., Z) = I(X, Y, ..., Z) - I(X, ..., Z) - I(Y, ..., Z) + I(..., Z)
        mi_K = mi_ksg(samples[:, K], k=k_ksg)
        # return np.maximum(0, mi_ijK - mi_iK - mi_jK + mi_K)
        return mi_ijK - mi_iK - mi_jK + mi_K


def mci_test(samples, i, j, K, alpha=0.05, k_ksg=15):
    """
    Multivariable conditional independence test

    :param samples: the collected samples
    :param i: the target node i
    :param j: the start node j
    :param K: conditional set K
    :param alpha: the significant level
    :param k_ksg: the k of knn in KSG method
    :return: whether satisfied conditional independence
    """
    if len(K) == 0:
        # # if K is empty, then implement granger causality test
        # gc = grangercausalitytests(samples[:, [i, j]], maxlag=1, verbose=True)
        # p_value = gc[1][0]['lrtest'][1]  # the p value of likelihood ratio test
        mi = mi_ksg(samples[:, [i, j]])

        if mi < alpha:
            return False, mi
        else:
            # conditional independence
            return True, mi

    # estimate I(i, j | K) by original data
    ce = causal_entropy(samples, i, j, K, k_ksg=k_ksg)

    # judge whether mci
    if ce < alpha:
        # print('i', i, 'j', j, 'K', K, cmi)
        # print('samples', samples[:10])
        # conditional independence
        return True, ce
    else:
        return False, ce
