import numpy as np
import statsmodels.tsa.stattools as sts
import netCDF4 as nc
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pk
import os
import time

from utils import *
from config import args

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
    # samples = samples[:, 1:]

    cn = np.zeros(shape=net.shape)
    for i in range(samples.shape[1]):
        for j in range(samples.shape[1]):
            ob = np.array([samples[:, i], samples[:, j]]).transpose([1, 0])
            results = sts.grangercausalitytests(ob, maxlag=5)

            lag = 1
            for _ in results.values():
                p_test = _[0]

                for __ in p_test.values():
                    cn[j, i, lag - 1] = 1 - __[1]

                lag += 1

            # print('ps ==', ps)

    theta = 0.7
    cn[cn < theta] = 0
    cn[cn >= theta] = 1

    # net = net[1:, 1:]
    tprs[l] = recall(net, cn)
    fprs[l] = false_positive_rate(net, cn)
    print('recall:\t', tprs[l])
    print('false positive rate:\t', fprs[l])

# save results
if not os.path.exists('./results/'):
    os.mkdir('./results/')

tpr_fpr = np.concatenate([tprs, fprs]).reshape([2, loop]).transpose([1, 0])
pd.DataFrame(tpr_fpr).to_csv(args.save_path, index=False, header=False)

# time
print('Time ==', time.time())
print('Time cost ==', time.time() - t0)
