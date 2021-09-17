from os.path import splitext
from os import listdir
import os
from os.path import join as opj
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import csv
import h5py
import matplotlib.pyplot as plt


class BasicDataset(Dataset):
    def __init__(self, dataset_pths, flag='train', mode='original'):
        self.dataset_pths = dataset_pths
        self.pths = []
        self.mode = mode
        for dataset_pth in dataset_pths:
            if flag == 'train':
                dataset_txt = opj(dataset_pth, 'trainset.txt')
                assert os.path.exists(dataset_txt) is True, 'There is no trainset.txt file.'
            elif flag == 'test':
                dataset_txt = opj(dataset_pth, 'testset.txt')
                assert os.path.exists(dataset_txt) is True, 'There is no testset.txt file.'
            else:
                raise ValueError('Param flag should only be "train" or "test".')

            with open(dataset_txt, 'r') as f:
                data = csv.reader(f, delimiter='\t')
                for line in data:
                    pth = opj(dataset_pth, 'data', line[0])
                    self.pths.append(pth)

        logging.info(f'Creating dataset with {len(self.pths)} examples')

    def __len__(self):
        return len(self.pths)

    def __getitem__(self, i):
        data_file_pth = self.pths[i]
        assert os.path.exists(data_file_pth) is True, f'There is no file: {data_file_pth}'

        file = h5py.File(data_file_pth, 'r')
        amplitude_input = np.array(file['input/amplitude'])
        phase_input = np.array(file['input/phase'])
        amplitude_target = np.array(file['output/amplitude'])
        phase_target = np.array(file['output/phase'])
        file.close()

        # normalization
        if self.mode == 'normal':
            phase_input = (phase_input + np.pi) / (2 * np.pi)
            phase_target = (phase_target + np.pi) / (2 * np.pi)

        # CHW
        amplitude_input = amplitude_input[np.newaxis, :, :]
        phase_input = phase_input[np.newaxis, :, :]
        amplitude_target = amplitude_target[np.newaxis, :, :]
        phase_target = phase_target[np.newaxis, :, :]

        return {
            'amplitude_input': torch.from_numpy(amplitude_input).type(torch.float),
            'phase_input': torch.from_numpy(phase_input).type(torch.float),
            'amplitude_target': torch.from_numpy(amplitude_target).type(torch.float),
            'phase_target': torch.from_numpy(phase_target).type(torch.float),
        }

