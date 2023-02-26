import argparse

parser = argparse.ArgumentParser()

# the significance level
parser.add_argument('--beta', default=0.05)

# the max time lag, namely tau value
parser.add_argument('--max_tau', default=5)

# the path of data file
parser.add_argument('--save_path', default='./results/N5_L2N_nonlinear.csv')

# parse arguments
args = parser.parse_args()
