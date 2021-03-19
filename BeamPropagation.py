#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Propagation of the Angular spectrum function
@author: Chengkun Li

My reference: W.Goodman's Introduction to Fourier Optics - 1996 version
"""

import numpy as np
import matplotlib.pyplot as plt


class Propagate():
    def __init__(self, img_complex, pixelsize, wavelength):
        """
        Implement of Angular Spectrum function, more detail can be fund in Goodman's book
        --Introduction to Fourier Optics.

        :param img_complex: np.array dtype=np.complex, this is the complex image of input
        :param pixelsize: float, image's real pixelsize size (mm is the unit)
        :param wavelengh: the beam's wavelength
        """
        self.wavelength = wavelength
        # the complex form of a wave, the Eq.3-11 of Goodman's book
        # the relationship between intensity and amplitude, the Eq.4-7 of Goodman's book
        self.U0 = img_complex.copy()
        self.Uz = None

        # sample frequency, reference: https://blog.csdn.net/tyfwin/article/details/89840956
        # remember y is the direction of axis-0
        h, w = self.U0.shape
        fx = sorted(np.fft.fftfreq(w, pixelsize))
        fy = sorted(np.fft.fftfreq(h, pixelsize))
        [self.fx, self.fy] = np.meshgrid(fx, fy)

    def AS(self, z):
        """Angular Spectrum function"""
        # A0, the Eq.3-58 of Goodman's book
        # Az, the Eq.3-66 of Goodman's book
        A0 = np.fft.fftshift(np.fft.fft2(self.U0))
        Az = (A0 * np.exp(1j * 2 * np.pi / self.wavelength *
                          np.sqrt((1 - self.wavelength**2 * (self.fx**2 + self.fy**2)).astype(np.complex)) * z))
        # Uz, the Eq.3-65 of Goodman's book
        self.Uz = np.fft.ifft2(np.fft.ifftshift(Az))
        # # intensity image of the propagate, the Eq.4-7 of Goodman's book
        # img = np.abs(self.Uz)**2
        # img = img / np.max(img)
        return self.Uz

    def AScirc(self, z):
        """Angular Spectrum function"""
        # A0, the Eq.3-58 of Goodman's book
        A0 = np.fft.fftshift(np.fft.fft2(self.U0))

        # circ functuion, the Eq.3-69 of Goodman's book
        circ = (self.wavelength**2 * (self.fx**2 + self.fy**2) < 1).astype(int)

        # Az, the Eq.3-66 of Goodman's book
        Az = (A0 * np.exp(1j * 2 * np.pi / self.wavelength * circ *
                          np.sqrt((1 - self.wavelength**2 * (self.fx**2 + self.fy**2)).astype(np.complex)) * z))
        # Uz, the Eq.3-65 of Goodman's book
        self.Uz = np.fft.ifft2(np.fft.ifftshift(Az))
        # # intensity image of the propagate, the Eq.4-7 of Goodman's book
        # img = np.abs(self.Uz)**2
        # img = img / np.max(img)
        return self.Uz


if __name__ == "__main__":
    """
    This part is to test the correction of my code.
    And the experiment 1 is to make a Fraunhofer Diffraction and see whether the result of 
    diffraction is right or not, and proof its correction by propagating back method.
    As for the experiment 2, we find the reasonable range of propagating distance.
    """
    # =====EXPERIMENT 1=====
    # the Gaussian beam
    sigma = 1
    x = np.linspace(-6 * sigma, 6 * sigma, 2 ** 8)
    y = np.linspace(-6 * sigma, 6 * sigma, 2 ** 8)
    X, Y = np.meshgrid(x, y, indexing='ij')
    img = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    h, w = img.shape
    pixelsize = np.abs(x[1] - x[0])

    # Fraunhofer Diffraction (Place a circle aperture in beam)
    img_circle = np.copy(img)
    img_circle[np.sqrt(X ** 2 + Y ** 2) > 0.5 * sigma] = 0
    img_circle_complex = img_circle * np.exp(1j * np.zeros_like(img_circle))
    img_circle = np.abs(img_circle_complex) ** 2

    # beam propagation
    PD = Propagate(img_circle_complex, pixelsize, 0.1)
    img_circle_complex_out = PD.AS(10)
    img_circle_out = np.abs(img_circle_complex_out) ** 2

    # Fraunhofer diffraction (Place a rectangle aperture in beam)
    radius = 8
    rec_area_h = [int(np.ceil(h / 2 - radius)), int(np.floor(h / 2 + radius))]
    rec_area_w = [int(np.ceil(w / 2 - radius)), int(np.floor(w / 2 + radius))]
    img_rectangle = np.zeros_like(img)
    img_rectangle[rec_area_h[0]: rec_area_h[1], rec_area_w[0]: rec_area_w[1]] = \
        img[rec_area_h[0]: rec_area_h[1], rec_area_w[0]: rec_area_w[1]]
    img_rectangle_complex = img_rectangle * np.exp(1j * np.zeros_like(img_rectangle))
    img_rectangle = np.abs(img_rectangle_complex) ** 2

    # beam propagation
    FD = Propagate(img_rectangle_complex, pixelsize, 0.1)
    img_rectangle_complex_out = FD.AS(0.00002)
    img_rectangle_out = np.abs(img_rectangle_complex_out) ** 2

    # propagate back
    FD_back = Propagate(img_rectangle_complex_out, pixelsize, 0.1)
    img_rectangle_complex_back = FD_back.AScirc(-0.00002)
    img_rectangle_back = np.abs(img_rectangle_complex_back) ** 2

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1); ax.imshow(img_circle); ax.set_title('Input')
    ax = fig.add_subplot(2, 2, 2); ax.imshow(img_circle_out); ax.set_title('Diffraction')
    ax = fig.add_subplot(2, 2, 3); ax.imshow(img_rectangle); ax.set_title('Input')
    ax = fig.add_subplot(2, 2, 4); ax.imshow(img_rectangle_out); ax.set_title('Diffraction')
    plt.suptitle('FRAUNHOFER DIFFRACTION')
    plt.tight_layout()

    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1); ax.imshow(img_rectangle); ax.set_title('Input')
    ax = fig.add_subplot(1, 3, 2); ax.imshow(img_rectangle_back); ax.set_title('Back')
    ax = fig.add_subplot(1, 3, 3); ax.imshow(np.abs(img_rectangle_complex - img_rectangle_back)); ax.set_title('Error')
    plt.suptitle('PROPAGATION BACK')
    plt.tight_layout()

    # =====EXPERIMENT 2=====
    img_rectangle_amplitude = np.abs(img_rectangle_complex)
    img_rectangle_phase = np.angle(img_rectangle_complex)

    zs = np.linspace(0, 1e12, 1000)
    MSEs = {'real': [], 'imag': [], 'amplitude': [], 'phase': []}
    for z in zs:
        FD = Propagate(img_rectangle_complex, pixelsize, 0.1)
        img_rectangle_complex_out = FD.AScirc(z)

        FD_back = Propagate(img_rectangle_complex_out, pixelsize, 0.1)
        img_rectangle_complex_back = FD.AScirc(-z)

        ErrorMatrix_complex = img_rectangle_complex - img_rectangle_complex_back
        ErrorMatrix_real = np.real(ErrorMatrix_complex)
        ErrorMatrix_imag = np.imag(ErrorMatrix_complex)
        MSE_real = np.square(ErrorMatrix_real).mean()
        MSE_imag = np.square(ErrorMatrix_imag).mean()
        MSEs['real'].append(MSE_real)
        MSEs['imag'].append(MSE_imag)

        img_rectangle_back_amplitude = np.abs(img_rectangle_complex_back)
        img_rectangle_back_phase = np.angle(img_rectangle_complex_back)
        ErrorMatrix_amplitude = img_rectangle_amplitude - img_rectangle_back_amplitude
        ErrorMatrix_phase = img_rectangle_phase - img_rectangle_back_phase
        MSE_amplitude = np.square(ErrorMatrix_amplitude).mean()
        MSE_phase = np.square(ErrorMatrix_phase).mean()
        MSEs['amplitude'].append(MSE_amplitude)
        MSEs['phase'].append(MSE_phase)

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(zs, MSEs['real'], label='real MSE')
    ax.plot(zs, MSEs['imag'], label='imag MSE')
    ax.set_xlabel('z'); ax.set_ylabel('MSE'); plt.legend()
    ax = fig.add_subplot(2, 1, 2)
    ax.plot(zs, MSEs['amplitude'], label='amplitude MSE')
    ax.plot(zs, MSEs['phase'], label='phase MSE')
    ax.set_xlabel('z'); ax.set_ylabel('MSE'); plt.legend()
    plt.suptitle('the error between input image and the propagation back image')
    plt.tight_layout()

    plt.show()




