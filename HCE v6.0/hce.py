import numpy as np
import pandas as pd
import multiprocessing as mp
import os
import time
import netCDF4 as nc

from config import args
from mci_test import mci_test
from utils import *


def CEMCI(i, samples, max_tau, shared_q):
    """
    Causal entropy based multi-variable conditional independence test algorithm

    :param i: the designated nodes i
    :param samples: the samples of all nodes
    :param max_tau: the max time lag, namely tau value
    :param shared_q: shared queue of multiprocess, which saves the results
    :return: Ni set, the set of real causal parents
             Ni_weights, its corresponding weights
    """

    #
    S = samples.shape[0]  # the number of samples
    V = samples.shape[1]  # the number of nodes
    Ni = []  # Ni set, the set of real causal parents

    # discover causal parents
    t0 = [[] for _ in range(S - max_tau)]  # the samples of other nodes
    t1 = samples.iloc[max_tau:, i].values.reshape([S - max_tau, 1])  # the samples of node i
    for tau in range(1, max_tau + 1):
        # generate samples by time lag, namely tau value
        # let t1 to the end of new samples with tau
        t0 = np.concatenate([t0, samples.iloc[max_tau - tau: S - tau, :].values], axis=1)
        st = np.concatenate([t0, t1], axis=1)

        # the corresponding indices
        K = Ni + [_ + V * (tau - 1) for _ in range(V)]
        I = V * tau

        # discover causal parents
        for j in K[-V:]:
            # conditional set K-j
            Kj = [_ for _ in K if _ != j]

            # multi-variable conditional independence test
            mci, p = mci_test(
                samples=st,
                i=I,
                j=j,
                K=Kj,
                alpha=args.alpha,
                k_ksg=args.k_ksg
            )

            if mci:
                K.remove(j)
            else:
                Ni.append(j)

    # remove weak connections
    st = np.concatenate([t0, t1], axis=1)
    Nii = []

    for j in Ni:
        K = [_ for _ in Ni if _ != j]

        # test multivariate conditional independence
        mci, p = mci_test(
            samples=st,
            i=st.shape[1] - 1,
            j=j,
            K=K,
            alpha=args.beta,
            k_ksg=args.k_ksg
        )

        if not mci:
            Nii.append(j)

    shared_q.put([i, Nii])


def HCE(samples):
    """
    Higher-order causal entropy algorithm,

    :param samples: the samples of all nodes
    :return: causal parents at multiple time lags
    """

    #
    V = samples.shape[1]  # the number of all nodes
    causal_network = np.zeros([V, V, args.max_tau])  # i -> j at t(+1) of [V, V, max_tau]

    # discover causal parents at multiple time lags
    # search causal parents based on CEMCI test
    results = mp.Manager().Queue()

    mp_pool = mp.Pool(processes=os.cpu_count() // 4 * 3)
    for i in range(V):
        mp_pool.apply_async(CEMCI, args=(i, samples, args.max_tau, results))

    mp_pool.close()
    mp_pool.join()

    # record results
    while not results.empty():
        result = results.get()

        for _, node in enumerate(result[1]):
            causal_network[node % V, result[0], node // V] = 1

    return causal_network


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
        # print(net)

        # run
        causal_network = HCE(samples)
        print_parents(causal_network)

        tprs[l] = recall(net, causal_network)
        fprs[l] = false_positive_rate(net, causal_network)
        print('recall:\t', tprs[l])
        print('false positive rate:\t', fprs[l])

    # save results
    if not os.path.exists('./results/'):
        os.mkdir('./results/')

    tpr_fpr = np.concatenate([tprs, fprs]).reshape([2, loop]).transpose([1, 0])
    pd.DataFrame(tpr_fpr).to_csv(args.save_path)

    # time
    print('Time ==', time.time())
    print('Time cost (s) ==', time.time() - t0)
