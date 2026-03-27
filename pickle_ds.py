import torch
from dataset import DataSet
import argparse
import os

SPLIT = (0.6, 0.2, 0.2)

parser = argparse.ArgumentParser()

parser.add_argument('--dimensions', nargs="*", type=int, default=[3, 3, 3],
                    help='Number of features for every perceptual dimension')
parser.add_argument('--game_size', type=int, default=10,
                    help='Number of target/distractor objects')
parser.add_argument('--sample_size', type=int, default=10,
                    help='Number of sample to get')
parser.add_argument('--scaling_factor', type=int, default=10,
                    help='Scaling factor for dataset generation.')
parser.add_argument("--save", type=bool, default=True)

args = parser.parse_args()

# prepare folder for saving
if not os.path.exists('data/'):
    os.makedirs('data/')

data_set = DataSet(args.dimensions,
                   game_size=args.game_size,
                   scaling_factor=args.scaling_factor,
                   device='cpu')
sample = args.sample_size
path = ('data/dim(' + str(len(args.dimensions)) + ',' + str(args.dimensions[0]) + ')' + str(sample) + '_sf' +
        str(args.scaling_factor) + '.ds')

if args.save:
    with open(path, "wb") as f:
        torch.save(data_set, f)
    print("Data set is saved as: " + path)
