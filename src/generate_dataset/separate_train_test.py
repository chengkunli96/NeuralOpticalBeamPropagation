"""
Separate the database into training and test sets according to a certain proportion
"""

import argparse
import os
from os.path import join as opj
import random


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Our generated dataset.')
    parser.add_argument('--testset_rate', type=float, dest='testset_rate', default=0.1,
                        help='The proportion of the test set in the overall dataset.')
    return parser.parse_args()


CURR_FILE_PATH = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    args = get_args()

    if not os.path.isabs(args.data_dir):
        args.data_dir = opj(CURR_FILE_PATH, args.data_dir)
    assert os.path.exists(args.data_dir) is True, \
        f'This dir does not exit: {args.data_dir}' \
        f'\n Please use a valid path.'
    dataset_pth = args.data_dir

    # get image paths
    files = os.listdir(opj(dataset_pth, 'data'))

    random.shuffle(files)
    testset_num_rate = args.testset_rate
    testset_num = int(testset_num_rate * len(files))
    testset = files[:testset_num]
    trainset = files[testset_num:]

    trainset_txt = opj(dataset_pth, 'trainset.txt')
    with open(trainset_txt, 'w') as f:
        for line in trainset:
            f.write(line)
            f.write('\n')

    testset_txt = opj(dataset_pth, 'testset.txt')
    with open(testset_txt, 'w') as f:
        for line in testset:
            f.write(line)
            f.write('\n')

    print('Finish !!!')