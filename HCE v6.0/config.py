"""
    Author      : kangmingyu
    Email       : kangmingyu.china@gmail.com
    Institute   : Southeast University, China
"""

import argparse

parser = argparse.ArgumentParser()

# k of knn in KSG method
parser.add_argument('--k_ksg', default=14)

# the significance level
parser.add_argument('--alpha', default=0.01)
parser.add_argument('--beta', default=0.02)

# the max time lag, namely tau value
parser.add_argument('--max_tau', default=5)

# the path of data file
parser.add_argument('--save_path', default='./results/N5_L2N_nonlinear.csv')

# parse arguments
args = parser.parse_args()
