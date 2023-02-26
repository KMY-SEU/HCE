import sys
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc

import seaborn as sns

sns.set_style('ticks')
sns.set_context(context='paper', font_scale=1.5)

import skccm as ccm

from skccm.utilities import train_test_split

# # data
# rx1 = 3.72  # determines chaotic behavior of the x1 series
# rx2 = 3.72  # determines chaotic behavior of the x2 series
# b12 = 0.2  # Influence of x1 on x2
# b21 = 0.01  # Influence of x2 on x1
# ts_length = 1000
# x1, x2 = data.coupled_logistic(rx1, rx2, b12, b21, ts_length)

data = nc.Dataset('./data/data0.nc', 'r')
samples, net = data['samples'][:], data['net'][:]

# # plot data
# plt.figure()
# fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
# ax[0].plot(x1[0:100])
# ax[1].plot(x2[0:100])
# ax[0].set_yticks([.1, .3, .5, .7, .9])
# ax[1].set_xlabel('Time')
# sns.despine()
# plt.show()

# embedding
e1 = ccm.Embed(samples[:, 0])
e2 = ccm.Embed(samples[:, 1])

# mi1 = e1.mutual_information(10)
# mi2 = e2.mutual_information(10)
#
# fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
# ax[0].plot(np.arange(1, 11), mi1)
# ax[1].plot(np.arange(1, 11), mi2)
# ax[1].set_xlabel('Lag')
# sns.despine()
# plt.show()

lag = 3
embed = 2
X1 = e1.embed_vectors_1d(lag, embed)
X2 = e2.embed_vectors_1d(lag, embed)

# fig, ax = plt.subplots(ncols=2, sharey=True, sharex=True, figsize=(10, 4))
# ax[0].scatter(X1[:, 0], X1[:, 1])
# ax[1].scatter(X2[:, 0], X2[:, 1])
# ax[0].set_xlabel('X1(t)')
# ax[0].set_ylabel('X1(t-1)')
# ax[1].set_xlabel('X2(t)')
# ax[1].set_ylabel('X2(t-1)')
# sns.despine()
# plt.show()

# CCM
CCM = ccm.CCM()

# split train and ccm with ratio 0.75
x1tr, x1te, x2tr, x2te = train_test_split(X1, X2, percent=.75)

print('x1tr ==', x1tr)
print('x1te ==', x1te)
print('x2tr ==', x2tr)
print('x2te ==', x2te)

len_tr = len(x1tr)
lib_lens = np.linspace(10, len_tr / 2, dtype='int')

CCM.fit(x1tr, x2tr)
x1p, x2p = CCM.predict(x1te, x2te, lib_lengths=lib_lens)

# %%

sc1, sc2 = CCM.score()
print('sc1:\t', sc1)
print('sc2:\t', sc2)

fig, ax = plt.subplots()
ax.plot(lib_lens, sc1, label='X1 xmap X2')
ax.plot(lib_lens, sc2, label='X2 xmap X1')
ax.set_xlabel('Library Length')
ax.set_ylabel('Forecast Skill')
sns.despine()
plt.show()
