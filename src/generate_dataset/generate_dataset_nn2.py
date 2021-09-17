"""
For this database, we use different images as the amplitude input and the phase input
respectively. And the tilted angle changes from 0 to 45 degree randomly.
"""

import numpy as np
import os
from os.path import join as opj
import cv2
import json
import random
import h5py
import sys
import argparse
import matplotlib.pyplot as plt

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from beam_propagate import BeamPropagation
from beam_propagate import nm, um, mm, cm, m
from get_image_paths import getImagePathes


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_dir', dest='original_dir', type=str, default='../../data/original/lfw',
                        help='The path of the original database')
    parser.add_argument('--out_dir', dest='out_dir', type=str, default='../../data/processed/lfw_nn2',
                        help='The path of the processed database')
    return parser.parse_args()


# get absolute path from flw dataset
CURR_FILE_PATH = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    args = get_args()

    if not os.path.isabs(args.original_dir):
        args.original_dir = opj(CURR_FILE_PATH, args.original_dir)
    if not os.path.isabs(args.out_dir):
        args.out_dir = opj(CURR_FILE_PATH, args.out_dir)
    assert os.path.exists(args.original_dir) is True, \
        f'This dir does not exit: {args.original_dir}' \
        f'\n Please use "--original_dir" to set it.'
    print(f'The original database that we are using is: {args.original_dir}')
    print(f'The output processed database is: {args.out_dir}')

    # save path and generate its subdirs
    savepath = opj(CURR_FILE_PATH, args.out_dir)
    datadir = 'data'
    savepath_datadir = opj(savepath, datadir)
    if not os.path.exists(savepath_datadir):
        os.makedirs(savepath_datadir)

    # beam propagation parameters (pre-defined)
    distance = 10 * mm
    pixelsize = (5 * um, 5 * um)  # w and h direction
    wavelength = 632.8 * nm
    # save beam propagation configuration
    config_dict = {
        'distance': distance,
        'pixelsize': pixelsize,
        'wavelength': wavelength,
        'tiltedangle': f'random from 0 to 45 degree',
    }
    json_str = json.dumps(config_dict, indent=4)
    with open(opj(savepath, 'configuration.json'), 'w') as json_file:
        json_file.write(json_str)

    # generate images from  the original database
    img_pths = getImagePathes(args.original_dir)
    print('The original database has been loading ...')

    # generate our database by the RAS algorithm
    for (n, img_pth) in enumerate(img_pths):
        tiltangle = random.randint(0, 45)
        tiltedangle = (0, tiltangle / 180 * np.pi)  # along x-axis, y-axis

        # get an image as the input amplitude image
        imgname = os.path.split(img_pth)[-1]
        imgname = os.path.splitext(imgname)[0]  # no post-suffix

        # amplitude input
        img = cv2.imread(img_pth, cv2.IMREAD_GRAYSCALE)  # shape is (250, 250)
        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        # normalize this amplitude image to range [0,1]
        img_nor = (img - np.min(img)) / (np.max(img) - np.min(img))
        amplitude = np.array(img_nor)

        # generate phase image from another human face image [-2pi, 2pi]
        num = random.randint(0, len(img_pths) - 1)
        while num == n:
            num = random.randint(0, len(img_pths) - 1)
        img2 = cv2.imread(img_pths[num], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.resize(img2, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        img2_nor = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))
        phase = 2 * np.pi * np.array(img2_nor) - np.pi

        # complex input data
        indata = amplitude * np.exp(1j * phase)
        BP = BeamPropagation(indata, pixelsize=pixelsize, wavelength=wavelength)

        # propagation and get the output images
        BP.setTiltedReferencePlane(tiltedangle[0], tiltedangle[1])
        BP.propagate(distance)
        amplitude_out = BP.getAmplitudeImage()
        phase_out = BP.getPhaseImage()

        # save as h5 file
        file_pth = opj(savepath_datadir, '{}.h5'.format(imgname))
        with h5py.File(file_pth, 'w') as f:
            dset1 = f.create_dataset('input/amplitude', data=amplitude)
            dset2 = f.create_dataset('input/phase', data=phase)
            dset3 = f.create_dataset('input/angle', data=tiltangle)
            dset4 = f.create_dataset('output/amplitude', data=amplitude_out)
            dset5 = f.create_dataset('output/phase', data=phase_out)

        # show the rate of progress
        rate = 100 * (n + 1) / len(img_pths)
        if len(img_pths) >= 100 and n % (len(img_pths) // 100) == 0:
            print('\rGenerating %.2f%% ...' % (rate), end='', flush=True)
        else:
            print('\rGenerating %.2f%% ...' % (rate), end='', flush=True)

        # # show for DEBUG
        # print(angle)
        # fig, axs = plt.subplots(2, 2)
        #
        # ax = axs[0, 0]
        # BP.plot(ax, 'input', 'amplitude')
        # ax.set_title('input Diffraction (amplitude)')
        #
        # ax = axs[1, 0]
        # BP.plot(ax, 'input', 'phase')
        # ax.set_title('input Diffraction (phase)')
        #
        # ax = axs[0, 1]
        # BP.plot(ax, 'output', 'amplitude')
        # ax.set_title('output Diffraction (amplitude)')
        #
        # ax = axs[1, 1]
        # BP.plot(ax, 'output', 'phase')
        # ax.set_title('output Diffraction (phase)')
        #
        # plt.show()

    print('\nGeneration finishes !!!')











