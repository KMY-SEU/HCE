import os
import time
import numpy as np
import pandas as pd
import netCDF4 as nc
import networkx as nx
import matplotlib.pyplot as plt

from causalnex.structure import StructureModel
from causalnex.structure.dynotears import from_pandas_dynamic
from utils import *


def print_parents(parents):
    """
    Print causal parents

    :param parents: causal parents, whose dimension is [n, n, max_lag]
    :return: None
    """

    for i in range(parents.shape[1]):
        print('\nnode', i, ':')

        for j in range(parents.shape[0]):
            time_lags = [str(-t - 1) + ' ' for t, _ in enumerate(parents[j, i]) if _]

            if time_lags:
                print(j, '->', i, ':', ''.join(time_lags))


if __name__ == '__main__':
    # time
    print('Time ==', time.time())
    t0 = time.time()

    # read data
    files = os.listdir('./data/')
    loop = len(files)
    tprs = np.zeros(shape=loop)
    fprs = np.zeros(shape=loop)
    for l, file in enumerate(files):
        print('loop ==', l)
        print('./data/' + file)

        data = nc.Dataset('./data/' + file, 'r')
        samples, net = data['samples'][:], data['net'][:]
        samples = pd.DataFrame(samples)

        # print(samples)
        # print(net.shape)

        N, _, P = net.shape

        sm = from_pandas_dynamic(samples, p=5, max_iter=100,
                                 lambda_w=0.05, lambda_a=0.05, w_threshold=0.01)

        causal_links = nx.adjacency_matrix(sm)
        dynamics_mtxs = causal_links.toarray()

        causal_network = dynamics_mtxs[N * 1: N * 2, N * 1: N * 2][:, :, np.newaxis]
        for p in range(2, P + 1):
            causal_network = np.concatenate([causal_network,
                                             dynamics_mtxs[N * p: N * (p + 1), N * p: N * (p + 1)][:, :, np.newaxis]],
                                            axis=2)

        causal_network[np.abs(causal_network) > 0] = 1.

        # print(causal_network)

        # run
        print_parents(causal_network)

        tprs[l] = recall(net, causal_network)
        fprs[l] = false_positive_rate(net, causal_network)
        print('recall:\t', tprs[l])
        print('false positive rate:\t', fprs[l])

    # save results
    if not os.path.exists('./results/'):
        os.mkdir('./results/')

    tpr_fpr = np.concatenate([tprs, fprs]).reshape([2, loop]).transpose([1, 0])
    pd.DataFrame(tpr_fpr).to_csv('./results/N6_L2N_nonlinear.csv', index=False, header=False)

    # time
    print('Time ==', time.time())
    print('Time cost ==', time.time() - t0)
