import numpy as np
import scipy.special as sps
from sklearn.neighbors import KDTree


def calculate_nvar_expectation(vars, k):
    """
    calculate the expectation of digamma values of nvars

    :param vars: the samples of n variables
    :param k: the super parameters of knn
    :return: an array recording the expectations
    """

    # dimension
    dims = vars.shape[1]

    # knn
    kdt = KDTree(vars, metric='euclidean')
    dist, indices = kdt.query(vars, k=k)

    # calculate expectation of digamma values of nvars
    nvars = np.zeros(shape=dims)

    for i in range(vars.shape[0]):
        for d in range(dims):
            # the radius (the epsilon in paper) of i-th sample on d-dim
            radius = np.max(np.abs(vars[indices[i], d] - vars[i, d]))

            # nvars
            nvar = np.sum(np.abs(vars[:, d] - vars[i, d]) < radius)
            nvars[d] += sps.digamma(nvar)

    nvars /= vars.shape[0]  # the expectation

    return nvars


def mi_ksg(vars, k=14):
    """
    calculate mutual information by KSG estimator

    :param vars: the samples of designated variables
    :param k: super parameter k of knn algorithm
    :return: mutual information of the variables
    """
    if vars.shape[1] < 2:
        print('error! the dimension of samples should be more than 1.')
        return -1

    N = vars.shape[0]  # the number of samples
    m = vars.shape[1]  # the number of variables

    # the expectation of digamma values of nvars
    nvars = calculate_nvar_expectation(vars, k=k)

    return np.maximum(0, sps.digamma(k) - (m - 1) / k + (m - 1) * sps.digamma(N) - np.sum(nvars))

########################测试代码##########################
#
# def mi_gauss(rho):
#     return -0.5 * np.log(1 - rho ** 2)
#
#
# rhos = [0, 0.3, 0.6, 0.9]
# ns_arr = np.arange(0, 10000, 100)
# errs = np.zeros([len(rhos), len(ns_arr)])
#
# for i in range(len(rhos)):
#     rho = rhos[i]
#
#     for j in range(1, len(ns_arr)):
#         ns = ns_arr[j]
#
#         cov = [[1, rho], [rho, 1]]
#         samples = np.random.multivariate_normal(mean=[0, 0], cov=cov, size=ns)
#
#         mi_ksg = mi_ksg(samples, k=15)
#         mi_g = mi_gauss(rho)
#
#         errs[i, j] = mi_ksg - mi_g
#
# import matplotlib.pyplot as plt
#
# plt.figure()
# for i in range(len(rhos)):
#     plt.plot(errs[i][::-1])
# plt.show()
#
##################################
#
# 最佳KSG参数, k = 14, ns > 200
#
####################################
