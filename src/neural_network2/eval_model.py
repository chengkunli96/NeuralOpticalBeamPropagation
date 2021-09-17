import argparse
import os
import numpy as np
from os.path import join as opj

import torch
import cv2
import matplotlib.pyplot as plt

from model.net.model import Net as Net
import imageio

# distance unit
m = 1.
cm = 1e-2
mm = 1e-3
um = 1e-6
nm = 1e-9

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--amplitude', required=True, type=str, help='amplitude input image')
    parser.add_argument('--phase', required=True, type=str, help='phase input image')
    parser.add_argument('--tilted_angle', required=True, type=float, help='the tilted angle along y-axis')
    parser.add_argument('--load', required=True, dest='load', type=str, help='Load model from a .pth file')
    parser.add_argument('--device', dest='device', type=int, default=0, help='gpu device')
    return parser.parse_args()


CURR_FILE_PATH = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    args = get_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    if args.device == -1:
        device = torch.device('cpu')
        print(f'Using device cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device {device} {args.device}')

    # ==== the pre-defined parameters of this model =====
    distance = 10 * mm
    pixelsize = (5 * um, 5 * um)  # (h, w) direction
    wavelength = 632.8 * nm
    tiltangle = args.tilted_angle
    tiltedangle = (0, tiltangle / 180 * np.pi)  # along x-axis, y-axis
    xtheta, ytheta = tiltedangle

    # ===== network initialization ======
    # set neural network
    net = Net(in_channels=2, out_channels=2)
    if os.path.isabs(args.load):
        model_file = args.load
    else:
        model_file = opj(CURR_FILE_PATH, args.load)
    net.load_state_dict(torch.load(model_file, map_location=device))
    print(f'Model loaded from {args.load}')
    net.eval()
    net.to(device)

    # ===== input data =====
    amplitude_input_pth = args.amplitude
    phase_input_pth = args.phase
    # amplitude input
    img = cv2.imread(amplitude_input_pth, cv2.IMREAD_GRAYSCALE)  # shape is (250, 250)
    img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    img_nor = (img - np.min(img)) / (np.max(img) - np.min(img))
    amplitude_input = np.array(img_nor)  # 0 - 1
    # phase input
    img2 = cv2.imread(phase_input_pth, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.resize(img2, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    img2_nor = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))
    phase_input = np.array(img2_nor).copy()
    # convert to torch (BCHW)
    amplitude_input = amplitude_input[np.newaxis, np.newaxis, :, :]
    phase_input = phase_input[np.newaxis, np.newaxis, :, :]
    amplitude_input = torch.from_numpy(amplitude_input).type(torch.float)
    phase_input = torch.from_numpy(phase_input).type(torch.float)
    # concatenate amplitude and phase
    input_data = torch.cat((amplitude_input, phase_input), 1)
    input_data = input_data.to(device=device, dtype=torch.float32)
    # angle input
    input_angle = torch.from_numpy([args.tilted_angle]).type(torch.float)
    input_angle = torch.unsqueeze(input_angle, 1)  # shape BC
    input_angle = input_angle.to(device=device, dtype=torch.float32)

    # ===== network prediction =====
    pred_data, _ = net(input_data, input_angle)
    amplitude_pred = pred_data.data[:, 0, :, :].numpy().squeeze()
    phase_pred = pred_data.data[:, 1, :, :].numpy().squeeze()

    # ===== rescale phase prediction from [0, 1] to [-pi, pi] =====
    phase_pred = 2 * np.pi * phase_pred - np.pi

    # ===== compensate the carrier frequency ====
    # rotation matrix
    rotate_trans_matrix_along_yaxis = np.array(
        [[np.cos(ytheta), 0, np.sin(ytheta)],
         [0, 1, 0],
         [-np.sin(ytheta), 0, np.cos(ytheta)]]
    )
    rotate_trans_matrix_along_xaxis = np.array(
        [[1, 0, 0],
         [0, np.cos(xtheta), -np.sin(xtheta)],
         [0, np.sin(xtheta), np.cos(xtheta)]]
    )
    trans_matrix = rotate_trans_matrix_along_yaxis @ rotate_trans_matrix_along_xaxis
    # carrier frequency
    u0, v0 = 0, 0
    w0 = np.sqrt(1 / wavelength ** 2 - u0 ** 2 - v0 ** 2)
    u_hat_0 = trans_matrix[0, 0] * u0 + trans_matrix[0, 1] * v0 + trans_matrix[0, 2] * w0
    v_hat_0 = trans_matrix[1, 0] * u0 + trans_matrix[1, 1] * v0 + trans_matrix[1, 2] * w0
    carrier_frequency_u = u_hat_0
    carrier_frequency_v = v_hat_0
    # spacial field (the image range transfers from Lw to Lw/cos(ytheta))
    h, w = amplitude_pred.shape
    h_hat = h
    w_hat = w
    pixelsize_w = (w * pixelsize[1] / np.cos(ytheta)) / w_hat
    pixelsize_h = (h * pixelsize[0] / np.cos(xtheta)) / h_hat
    x_list = np.array(sorted(np.fft.fftfreq(w_hat))) * w_hat * pixelsize[1]
    y_list = np.array(sorted(np.fft.fftfreq(h_hat))) * h_hat * pixelsize[0]
    [x_hat, y_hat] = np.meshgrid(x_list, y_list)
    # compensate this factor
    f = amplitude_pred * np.exp(1j * phase_pred)
    f = f * np.exp(1j * 2 * np.pi * carrier_frequency_u * x_hat) \
        * np.exp(1j * 2 * np.pi * carrier_frequency_v * y_hat)
    phase_pred = np.angle(f)

    # ===== resize the output =====
    rate_h = 1 / np.cos(xtheta)
    new_h = int(np.round(rate_h * amplitude_pred.shape[0]))
    rate_w = 1 / np.cos(ytheta)
    new_w = int(np.round(rate_w * amplitude_pred.shape[1]))
    amplitude_pred = cv2.resize(amplitude_pred, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    phase_pred = cv2.resize(phase_pred, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # ===== show in figure ====
    fig, axs = plt.subplots(2, 2)

    h_length = h * pixelsize[0]
    w_length = w * pixelsize[1]

    ax = axs[0, 0]
    ax.imshow(amplitude_input.numpy().squeeze(),
              extent=[-w_length / 2 / mm, w_length / 2 / mm, -h_length / 2 / mm, h_length / 2 / mm, ])
    ax.set_title('amplitude input')

    ax = axs[1, 0]
    ax.imshow(phase_input.numpy().squeeze(),
              extent=[-w_length / 2 / mm, w_length / 2 / mm, -h_length / 2 / mm, h_length / 2 / mm, ])
    ax.set_title('amplitude input')

    h_length = new_h * pixelsize[0]
    w_length = new_w * pixelsize[1]

    ax = axs[0, 1]
    ax.imshow(amplitude_pred,
              extent=[-w_length / 2 / mm, w_length / 2 / mm, -h_length / 2 / mm, h_length / 2 / mm, ])
    ax.set_title('amplitude prediction')

    ax = axs[1, 1]
    ax.imshow(phase_pred,
              extent=[-w_length / 2 / mm, w_length / 2 / mm, -h_length / 2 / mm, h_length / 2 / mm, ])
    ax.set_title('phase prediction')

    plt.show()

