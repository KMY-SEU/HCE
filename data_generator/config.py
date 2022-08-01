"""
    Author      : kangmingyu
    Email       : kangmingyu.china@gmail.com
    Institute   : Southeast University, China
"""

import argparse

parser = argparse.ArgumentParser()

# the max time lag, namely tau value
parser.add_argument('--max_tau', default=5)

# the test times
parser.add_argument('--times', default=20)

# data length
parser.add_argument('--sample_size', default=2000)

# the path of data file
parser.add_argument('--save_path', default='./data/N5_L2N_nonlinear/')

# parse arguments
args = parser.parse_args()
