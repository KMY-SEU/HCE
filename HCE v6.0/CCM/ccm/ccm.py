import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
import pickle as pk
import seaborn as sns
import os
import time

sns.set_style('ticks')
sns.set_context(context='paper', font_scale=1.5)

import skccm as ccm

from skccm.utilities import train_test_split
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

    embed = 2
    cn = np.zeros(shape=net.shape)
    for i in range(samples.shape[1]):
        for j in range(samples.shape[1]):
            for lag in range(1, 6):
                # embedding
                e1 = ccm.Embed(samples[:, i])
                e2 = ccm.Embed(samples[:, j])

                X1 = e1.embed_vectors_1d(lag, embed)
                X2 = e2.embed_vectors_1d(lag, embed)

                # CCM
                CCM = ccm.CCM()

                # split train and ccm with ratio 0.75
                x1tr, x1te, x2tr, x2te = train_test_split(X1, X2, percent=.75)

                len_tr = len(x1tr)
                lib_lens = np.linspace(10, len_tr / 2, dtype='int')

                CCM.fit(x1tr, x2tr)
                x1p, x2p = CCM.predict(x1te, x2te, lib_lengths=lib_lens)

                # predict
                sc1, sc2 = CCM.score()
                # print('sc1:\t', sc1)
                # print('sc2:\t', sc2)

                # fig, ax = plt.subplots()
                # ax.plot(lib_lens, sc1, label='X1 xmap X2')
                # ax.plot(lib_lens, sc2, label='X2 xmap X1')
                # ax.set_xlabel('Library Length')
                # ax.set_ylabel('Forecast Skill')
                # ax.legend(['X1 xmap X2', 'X2 xmap X1'])
                # sns.despine()
                # plt.show()

                # sc2_1 = [sc2[_] - sc1[_] for _ in range(len(sc1))]
                sc2_1 = [sc2[_] for _ in range(len(sc1))]
                sc2_1.append(0)
                cn[j, i, lag - 1] = np.max(sc2_1)

    _max_ccm = np.max(cn)
    _min_ccm = np.min(cn)
    cn = (cn - _min_ccm) / (_max_ccm - _min_ccm)

    theta = 0.8
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
