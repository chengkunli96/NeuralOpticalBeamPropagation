
# -*- coding: utf-8 -*-

"""
Propagation of the Angular spectrum function
@author: Chengkun Li

My reference:
Angular spectrum function: W.Goodman's Introduction to Fourier Optics - 1996 version
"""

import numpy as np
import matplotlib.pyplot as plt
from diffractsim import colour_functions as cf
from scipy import interpolate

# length units
m = 1.
cm = 1e-2
mm = 1e-3
um = 1e-6
nm = 1e-9


class BeamPropagation(object):
    """
    Computing the propagation of mono-chromatic beam by Angular Spectrum function,
    more detail can be fund in Goodman's book -- Introduction to Fourier Optics.
    """

    def __init__(self, img_input, pixelsize, wavelength, padding=True):
        """
        Initialize the source complex wave filed, representing the cross-section profile of a plane wave.

        :param img_input: np.array dtype=np.complex,
            the input image which is the the cross-section profile of the plane wave
        :param pixelsize: tuple (width, height), image's real pixelsize size (mm is the unit)
        :param wavelengh: the wavelength of the plane wave
        """
        self.wavelength = wavelength
        self.pixelsize = pixelsize
        self.padding = padding
        # the complex form of a wave, the Eq.3-11 of Goodman's book
        # the relationship between intensity and amplitude, the Eq.4-7 of Goodman's book
        h, w = img_input.shape
        self.input = img_input

        # padding
        if padding:
            Nypad = int(np.floor(h / 2))
            Nxpad = int(np.floor(w / 2))
            self.Nypad = Nypad
            self.Nxpad = Nxpad
            img_input = np.pad(img_input, ((Nypad, Nypad), (Nxpad, Nxpad)), 'constant')
            h, w = img_input.shape

        # initializing
        self.U0 = img_input.copy().astype(np.complex)
        self.Uz = img_input.copy().astype(np.complex)
        self.I0 = np.abs(self.U0) ** 2
        self.Iz = np.abs(self.Uz) ** 2

        # sample frequency (frequency field), reference:
        # https://www.physicsforums.com/threads/spatial-frequency-of-pixels-in-an-fft-transformed-image.679406/
        # remember y is the direction of axis-0
        h, w = self.U0.shape
        fx_list = sorted(np.fft.fftfreq(w, pixelsize[0]))
        fy_list = sorted(np.fft.fftfreq(h, pixelsize[1]))
        [self.fx, self.fy] = np.meshgrid(fx_list, fy_list)

        # spacial field
        x_list = np.array(sorted(np.fft.fftfreq(w))) * w * pixelsize[0]
        y_list = np.array(sorted(np.fft.fftfreq(h))) * h * pixelsize[1]
        [self.x, self.y] = np.meshgrid(x_list, y_list)

        # set reference plane's tilted angle
        self.reference_plane_is_tilted = False
        self.ytheta = 0
        self.xtheta = 0
        self.trans_matrix = np.eye(3)  # transfer the  source coord to the reference coord

    def setTiltedReferencePlane(self, xtheta, ytheta, carrier_frequency_flag=False):
        """
        Compute spacial frequency and jacobian matrix for diffraction on the tilted plane.
        My reference -- Fast calculation method for optical diffraction on tilted planes
        by use of the angular spectrum of plane waves.

        :param xtheta: np.float, range(-0.5pi -> +0.5pi), the angle rotating x-axis.
        :param ytheta: np.float, range(-0.5pi -> +0.5pi), the angle rotating y-axis.
        """
        # if this flag is true, the we compensate this factor. Else, we eliminate it.
        self.carrier_frequency_flag = carrier_frequency_flag

        # a matrix used to transform the source coordinates into the reference coordinates (inverse transform
        # means a contrary transformation direction)
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
        rotate_trans_matrix = rotate_trans_matrix_along_yaxis @ rotate_trans_matrix_along_xaxis

        self.ytheta = ytheta
        self.xtheta = xtheta
        self.trans_matrix = rotate_trans_matrix
        self.reference_plane_is_tilted = True

    def fastRAS(self, z, carrier_frequency_flag=False):
        """
        The fast rotated angular spectrum (RAS). My reference -- Fast calculation method for
        optical diffraction on tilted planes by use of the angular spectrum of plane waves.
        """
        G0 = np.fft.fftshift(np.fft.fft2(self.U0))  # Eq.1 of my reference
        Gd = (
            G0 * np.exp(1j * 2 * np.pi / self.wavelength *
            np.sqrt((1 - (self.wavelength) ** 2 * (self.fx ** 2 + self.fy ** 2)).astype(np.complex)) * z)
        )  # Eq.20 of my reference
        F = Gd  # Eq.22 of the reference

        # transformation matrix
        trans_matrix = self.trans_matrix
        inv_trans_matrix = np.linalg.inv(trans_matrix)

        # the spacial frequency on the source coordinates. Eq.6 of the reference
        u = self.fx.copy()
        v = self.fy.copy()
        w = np.sqrt(1 / self.wavelength ** 2 - u ** 2 - v ** 2)

        # find the carrier frequency
        u0, v0 = 0, 0
        w0 = np.sqrt(1 / self.wavelength ** 2 - u0 ** 2 - v0 ** 2)
        u_hat_0 = trans_matrix[0, 0] * u0 + trans_matrix[0, 1] * v0 + trans_matrix[0, 2] * w0
        v_hat_0 = trans_matrix[1, 0] * u0 + trans_matrix[1, 1] * v0 + trans_matrix[1, 2] * w0
        w_hat_0 = trans_matrix[2, 0] * u0 + trans_matrix[2, 1] * v0 + trans_matrix[2, 2] * w0

        # the wave vector transformation. the Eq.9 of the reference. (non-uniform grid)
        u_hat = trans_matrix[0, 0] * u + trans_matrix[0, 1] * v + trans_matrix[0, 2] * w
        v_hat = trans_matrix[1, 0] * u + trans_matrix[1, 1] * v + trans_matrix[1, 2] * w
        w_hat = trans_matrix[2, 0] * u + trans_matrix[2, 1] * v + trans_matrix[2, 2] * w

        # find the uniform spectrum grid on the reference plane.
        h, w = self.U0.shape
        h_hat = h
        # w_hat = int(w / np.cos(self.ytheta))  # w
        w_hat = w
        # the image range transfers from Lw to Lw/cos(ytheta)
        pixelsize_w = (w * self.pixelsize[0] / np.cos(self.ytheta)) / w_hat
        pixelsize_h = (h * self.pixelsize[1] / np.cos(self.xtheta)) / h_hat
        u_list = sorted(np.fft.fftfreq(w_hat, pixelsize_w))
        v_list = sorted(np.fft.fftfreq(h_hat, pixelsize_h))
        [u_hat_grid, v_hat_grid] = np.meshgrid(u_list, v_list)
        u_hat_grid = u_hat_grid + u_hat_0
        v_hat_grid = v_hat_grid + v_hat_0
        w_hat_grid = np.sqrt(1 / self.wavelength ** 2 - u_hat_grid ** 2 - v_hat_grid ** 2)

        # interpolation (cubic spline)
        u_interp = inv_trans_matrix[0, 0] * u_hat_grid + inv_trans_matrix[0, 1] * v_hat_grid + inv_trans_matrix[0, 2] * w_hat_grid
        v_interp = inv_trans_matrix[1, 0] * u_hat_grid + inv_trans_matrix[1, 1] * v_hat_grid + inv_trans_matrix[1, 2] * w_hat_grid
        F_grid = interpolate.griddata(
            np.array([v.ravel(), u.ravel()]).T,
            F.ravel(),
            np.array([v_interp.ravel(), u_interp.ravel()]).T,
            method='cubic',
        )
        F_grid = F_grid.reshape(u_hat_grid.shape[0], u_hat_grid.shape[1])
        F_grid = np.nan_to_num(F_grid)

        # update the spectrum which changes from the non-uniform grid to the uniform grid
        F = F_grid
        u_hat = u_hat_grid
        v_hat = v_hat_grid

        # eliminate carrier frequency, Eq.26 of the reference
        carrier_frequency_u = u_hat_0
        carrier_frequency_v = v_hat_0
        u_dot = u_hat - carrier_frequency_u
        v_dot = v_hat - carrier_frequency_v
        w_dot = np.sqrt(1 / self.wavelength ** 2 - u_dot ** 2 - v_dot ** 2)

        # the jacobian matrix. Eq.14 of the reference.
        jacobian = (
            (inv_trans_matrix[0, 1] * inv_trans_matrix[1, 2]
             - inv_trans_matrix[0, 2] * inv_trans_matrix[1, 1]) * u_dot / w_dot
            + (inv_trans_matrix[0, 2] * inv_trans_matrix[1, 0]
               - inv_trans_matrix[0, 0] * inv_trans_matrix[1, 2]) * v_dot / w_dot
            + (inv_trans_matrix[0, 0] * inv_trans_matrix[1, 1] - inv_trans_matrix[0, 1] * inv_trans_matrix[1, 0])
        )

        # spacial field
        x_list = np.array(sorted(np.fft.fftfreq(w_hat))) * w_hat * pixelsize_w
        y_list = np.array(sorted(np.fft.fftfreq(h_hat))) * h_hat * pixelsize_h
        [self.x_hat, self.y_hat] = np.meshgrid(x_list, y_list)

        # Do FFT. Eq.16 of the reference.
        f = np.fft.ifft2(np.fft.ifftshift(F * np.abs(jacobian)))
        if carrier_frequency_flag:
            f = f * np.exp(1j * 2 * np.pi * carrier_frequency_u * self.x_hat) \
                * np.exp(1j * 2 * np.pi * carrier_frequency_v * self.y_hat)

        self.Uz = f
        # intensity image of the propagate, the Eq.4-7 of Goodman's book
        self.Iz = np.abs(f) ** 2

    def numericalSimulationRAS(self, z, carrier_frequency_flag=False):
        """
        The numerical simulation for only one-axis rotation which mentioned in the Fig.5 of
        the paper 'Fast calculation method for optical diffraction on tilted planes
        by use of the angular spectrum of plane waves'.
        """
        assert self.xtheta == 0 or self.ytheta ==0, 'This method is only used for one-axis rotation case.'
        G0 = np.fft.fftshift(np.fft.fft2(self.U0))  # Eq.1 of my reference
        Gd = (
                G0 * np.exp(1j * 2 * np.pi / self.wavelength *
                            np.sqrt((1 - self.wavelength ** 2 * (self.fx ** 2 + self.fy ** 2)).astype(np.complex)) * z)
        )  # Eq.20 of my reference
        F = Gd  # Eq.22 of the reference

        # transformation matrix
        trans_matrix = self.trans_matrix
        inv_trans_matrix = np.linalg.inv(trans_matrix)

        # find the carrier frequency
        u0 = 0
        v0 = 0
        w0 = np.sqrt(1 / self.wavelength ** 2 - u0 ** 2 - v0 ** 2)
        u_hat_0 = trans_matrix[0, 0] * u0 + trans_matrix[0, 1] * v0 + trans_matrix[0, 2] * w0
        v_hat_0 = trans_matrix[1, 0] * u0 + trans_matrix[1, 1] * v0 + trans_matrix[1, 2] * w0
        carrier_frequency_u = u_hat_0

        # the spacial frequency on the source coordinates. Eq.6 of the reference
        u = self.fx.copy()
        v = self.fy.copy()
        w = np.sqrt(1 / self.wavelength ** 2 - u ** 2 - v ** 2)

        # propagation distance z list
        if self.ytheta != 0:
            l = self.U0.shape[1] - 1
            interval = self.pixelsize[0] * np.tan(self.ytheta)
        else:
            l = self.U0.shape[0] - 1
            interval = self.pixelsize[1] * np.tan(self.xtheta)
        z_list = np.arange(z - np.ceil(l / 2) * interval, z + np.floor(l / 2) * interval, interval)

        f = np.zeros_like(self.U0).astype(np.complex)
        for (num, distance) in enumerate(z_list):
            self._AS(distance)
            if self.ytheta != 0:
                f[:, num] = self.Uz[:, num]
            else:
                f[num, :] = self.Uz[num, :]

        self.x_hat = self.x / np.cos(self.ytheta)
        self.y_hat = self.y / np.cos(self.xtheta)

        # if the carrier frequency is eliminated
        if not carrier_frequency_flag:
            f = f * np.exp(-1j * 2 * np.pi * carrier_frequency_u * self.x_hat)

        self.Uz = f
        # intensity image of the propagate, the Eq.4-7 of Goodman's book
        self.Iz = np.abs(f) ** 2

        if self.padding:
            h, w = self.Uz.shape
            Uz = self.Uz[self.Nypad:-self.Nypad, self.Nxpad:-self.Nxpad]
            self.Uz = Uz
            self.Iz = np.abs(self.Uz) ** 2


    def AS(self, z):
        """
        Angular Spectrum (AS) function
        :param z: the propagation distance
        """
        # A0, the Eq.3-58 of Goodman's book
        # Az, the Eq.3-66 of Goodman's book
        A0 = np.fft.fftshift(np.fft.fft2(self.U0))
        Az = (
            A0 * np.exp(1j * 2 * np.pi / self.wavelength *
            np.sqrt((1 - self.wavelength ** 2 * (self.fx ** 2 + self.fy ** 2)).astype(np.complex)) * z)
        )

        # Uz, the Eq.3-65 of Goodman's book
        self.Uz = np.fft.ifft2(np.fft.ifftshift(Az))

        # intensity image of the propagate, the Eq.4-7 of Goodman's book
        self.Iz = np.abs(self.Uz) ** 2

    def ASback(self, z):
        """
        Angular Spectrum function with circ function which is shown in the Eq.3-69 of Goodman's book
        Do the back propagation for verifying the correction of our angular spectrum function
        :param z: the propagation distance
        """
        # A0, the Eq.3-58 of Goodman's book
        A0 = np.fft.fftshift(np.fft.fft2(self.U0))

        # circ functuion, the Eq.3-69 of Goodman's book
        circ = (self.wavelength**2 * (self.fx**2 + self.fy**2) < 1).astype(int)

        # Az, the Eq.3-66 of Goodman's book
        Az = (
            A0 * np.exp(1j * 2 * np.pi / self.wavelength * circ *
            np.sqrt((1 - self.wavelength**2 * (self.fx**2 + self.fy**2)).astype(np.complex)) * z)
        )
        # Uz, the Eq.3-65 of Goodman's book
        self.Uz = np.fft.ifft2(np.fft.ifftshift(Az))
        # # intensity image of the propagate, the Eq.4-7 of Goodman's book
        self.Iz = np.abs(self.Uz) ** 2

    def propagate(self, z, mode='propagation'):
        """
        :param z: the propagation distance
        """
        if not self.reference_plane_is_tilted:
            if z >= 0:
                self.AS(z)
            else:
                self.ASback(z)
        else:
            if z >= 0:
                if mode == 'numerical_simulation':
                    self.numericalSimulationRAS(z,
                        carrier_frequency_flag=self.carrier_frequency_flag)
                else:
                    self.fastRAS(z,
                        carrier_frequency_flag=self.carrier_frequency_flag)

            else:
                raise ValueError('We did not implement the back propagation '
                                 'on the tilted planes.')
        if self.padding:
            h, w = self.Uz.shape
            Uz = self.Uz[self.Nypad:-self.Nypad, self.Nxpad:-self.Nxpad]
            self.Uz = Uz
            self.Iz = np.abs(self.Uz) ** 2

    def getComplexImage(self, flag='output'):
        if flag == 'input':
            return self.input
        elif flag == 'output':
            return self.Uz
        else:
            raise ValueError('Parameter flag should only be "input" or "output".')

    def getPhaseImage(self, flag='output'):
        if flag == 'input':
            phase_img = np.angle(self.input)
            return phase_img
        elif flag == 'output':
            phase_img = np.angle(self.Uz)
            return phase_img

    def getAmplitudeImage(self, flag='output'):
        if flag == 'input':
            amplitude_img = np.abs(self.input)
            return amplitude_img
        elif flag == 'output':
            amplitude_img = np.abs(self.Uz)
            return amplitude_img

    def getIntensityImage(self, flag='output'):
        if flag == 'input':
            return np.abs(self.input) ** 2
        elif flag == 'output':
            return self.Iz
        else:
            raise ValueError('Parameter flag should only be "input" or "output".')

    def getRgbImage(self, flag='output'):
        """Colored the intensity image with the corresponding beam which has a certain wavelength."""
        if flag == 'input':
            I = np.abs(self.input) ** 2
        elif flag == 'output':
            I = self.Iz
        else:
            raise ValueError('Parameter flag should only be "input" or "output".')
        rgb = cf.wavelength_to_sRGB(self.wavelength / nm, 10 * I.flatten()).T
        rgb = rgb.reshape((I.shape[0], I.shape[1], 3))  # image's size can't change
        return rgb

    def plot(self, ax, flag='input', mode='intensity'):
        # plt.style.use("dark_background")
        if flag == 'input':
            if mode == 'intensity':
                img = self.getRgbImage('input')
                vmin = None
                vmax = None
            elif mode == 'amplitude':
                img = self.getAmplitudeImage('input')
                vmin = None
                vmax = None
            elif mode == 'phase':
                img = self.getPhaseImage('input')
                vmin = -np.pi / 2
                vmax = np.pi / 2
            h = img.shape[0]
            w = img.shape[1]
            h_length = h * self.pixelsize[1]
            w_length = w * self.pixelsize[0]

            im = ax.imshow(
                img,
                extent=[-w_length / 2 / mm, w_length / 2 / mm,
                        -h_length / 2 / mm, h_length / 2 / mm,],
                interpolation="spline36",
                vmin=vmin, vmax=vmax
            )
            ax.set_aspect((h_length / h) / (w_length / w))

            ax.set_xlabel('[mm]')
            ax.set_ylabel('[mm]')
        elif flag == 'output':
            if mode == 'intensity':
                img = self.getRgbImage('output')
                vmin = None
                vmax = None
            elif mode == 'amplitude':
                img = self.getAmplitudeImage('output')
                vmin = None
                vmax = None
            elif mode == 'phase':
                img = self.getPhaseImage('output')
                vmin = -np.pi
                vmax = np.pi
            h = img.shape[0]
            w = img.shape[1]
            if not self.reference_plane_is_tilted:
                h_length = h * self.pixelsize[1]
                w_length = w * self.pixelsize[0]

                im = ax.imshow(
                    img,
                    extent=[-w_length / 2 / mm, w_length / 2 / mm,
                            -h_length / 2 / mm, h_length / 2 / mm, ],
                    # interpolation="spline36",
                    vmin=vmin, vmax=vmax
                )
                ax.set_aspect('equal')
                ax.set_xlabel('[mm]')
                ax.set_ylabel('[mm]')
            else:
                im = ax.imshow(
                    img,
                    extent=[np.min(self.x_hat) / mm, np.max(self.x_hat) / mm,
                            np.min(self.y_hat) / mm, np.max(self.y_hat) / mm, ],
                    # interpolation="spline36",
                    vmin=vmin, vmax=vmax,
                )
                ax.set_aspect('equal')
                ax.set_xlabel('[mm]')
                ax.set_ylabel('[mm]')
        else:
            raise ValueError('Parameter flag should only be "input" or "output".')
        return im





