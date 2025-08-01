# -*- coding: utf-8 -*-
"""FFTLOG MODULE

Class to perform integrals with the FFTLog algorithm.
Class to implement the  Fast Fourier Transform in logarithmic space (FFTLog)
algorithm to perform the Fourier transform of a periodic sequence of
logarithmically spaced points. This is based on the work published in
`Hamilton (2000) <https://arxiv.org/abs/astro-ph/9905191v4>` and
`Fang et al. (2020) <https://arxiv.org/abs/1911.11947>`
"""

import numpy as np
from numpy.fft import rfft, irfft
import cloe.fftlog.utils as utils


class fftlog(object):
    """Class for the FFTLog algorithm.

    It computes integrals involving a Bessel function with the algorithm
    described in
    `Hamilton (2000) <https://arxiv.org/abs/astro-ph/9905191v4>`__ .
    It decomposes the integrand function using the Fast Fourier Transform;
    after this decomposition, each term can be integrated analytically.
    """

    def __init__(self, x, fx, nu=1.1, N_extrap_begin=0, N_extrap_end=0,
                 c_window_width=0.25, N_pad=0):
        """List of parameters.

        Parameters
        ----------
        x: numpy.ndarray
            Original logarithmically sampled domain of the input function. If
            not even, the algorithmic implementation modifies it accordingly
        fx: numpy.ndarray
            Original sampled input function
        nu: float
            Bias index, used to have a more stable fast Fourier transform
        N_extrap_begin: int
            Number of extrapolated points at the beginning of input arrays
        N_extrap_end: int
            Number of extrapolated points at the end of input arrays
        c_window_width: float
            Fraction of the ``c_m`` coefficients smoothed by the window
        N_pad: int
            Number of zero-padded points at the beginning and end of ``fx``
            array
        """
        self.x_original = x
        self.dlnx = np.log(x[1] / x[0])
        self.fx_original = fx
        self.nu = nu
        self.N_extrap_begin = N_extrap_begin
        self.N_extrap_end = N_extrap_end
        self.c_window_width = c_window_width

        # extrapolate x and f(x) linearly in log(x), and log(f(x))
        self.x = utils._log_extrap(x, N_extrap_begin, N_extrap_end)
        self.fx = utils._log_extrap(fx, N_extrap_begin, N_extrap_end)
        self.N = self.x.size  # length of the array after manipulations

        # zero-padding
        if (N_pad):
            pad = np.zeros(N_pad)
            self.x = utils._log_extrap(self.x, N_pad, N_pad)
            self.fx = np.hstack((pad, self.fx, pad))
            self.N += 2 * N_pad
            self.N_extrap_end += N_pad
            self.N_extrap_begin += N_pad  # update after padding

        if (self.N % 2 == 1):  # force they array size to be even, as
            self.x = self.x[:-1]  # required by the algorithm
            self.fx = self.fx[:-1]
            self.N -= 1
            if (N_extrap_end):
                self.N_extrap_end -= 1

        self.m, self.c_m = self._get_c_m()
        # array of eta_m, the exponent in the power law decomposition
        self.eta_m = 2 * np.pi * self.m / (float(self.N) * self.dlnx)

    def _get_c_m(self):
        r"""Gets smoothed coefficients.

        Computes the smoothed FFT coefficients of "biased" input
        function f(x): f_biased = f(x) / x^\nu
        (number of x values should be even).

        Returns
        -------
        Indexes m: array
            Indexes of the FFTW decomposition
        Coefficients c_m: array
            Smoothed coefficients on the biased input function
        """
        f_biased = self.fx * self.x ** (-self.nu)
        # biased f(x) array, to improve numerical stability
        c_m = rfft(f_biased)  # coefficients of the power-law decomposition
        m = np.arange(0, self.N // 2 + 1)
        c_m = c_m * utils._c_window(m, int(self.c_window_width * self.N // 2))
        return m, c_m

    def fftlog(self, ell):
        r"""Performs the fftlog.

        Calculates

        .. math::
            F(y) = \int_0^\infty \frac{dx}{x} f(x) j_{\rm \ell}(xy)

        where :math:`j_\ell` is the spherical Bessel function of order
        :math:`\ell`.

        Parameters
        ----------
        ell: int
            Order of the Bessel function present in the integral

        Returns
        -------
        y, Fy: numpy.ndarray, numpy.ndarray
         Logarithmically spaced array, set as
         ``y[:] = (ell+1)/x[::-1]``, FFTlog transform evaluated
         over the ``y`` array
        """
        z_ar = self.nu + 1j * self.eta_m
        y = (ell + 1.) / self.x[::-1]
        # TODO: possible improvement. y can be evaluated once and stored
        h_m = self.c_m * (self.x[0] * y[0])**(-1j * self.eta_m) \
            * utils._g_l(ell, z_ar)
        # TODO: possible improvement. _g_l can be evaluated once and stored

        Fy = irfft(np.conj(h_m)) * y**(-self.nu) * np.sqrt(np.pi) / 4.
        # here the ordering of N_extrap_begin and N_extrap_end is reversed
        # since we have moved to Fourier space
        return (
            y[self.N_extrap_end:self.N - self.N_extrap_begin],
            Fy[self.N_extrap_end:self.N - self.N_extrap_begin]
        )

    def fftlog_first_derivative(self, ell):
        r"""Performs the fftlog with the first derivative of the Bessel.

        Calculates

        .. math::
            F(y) = \int_0^\infty \frac{dx}{x} f(x) j'_{\rm \ell}(xy)

        where :math:`j_\ell` is the spherical Bessel function of order
        :math:`\ell`.

        Parameters
        ----------
        ell: int
            Order of the Bessel function present in the integral

        Returns
        -------
        y, Fy: numpy.ndarray, numpy.ndarray
         Logarithmically spaced array, set as
         ``y[:] = (ell+1)/x[::-1]``, FFTlog transform evaluated
         over the ``y`` array
        """
        z_ar = self.nu + 1j * self.eta_m
        y = (ell + 1.) / self.x[::-1]
        # TODO: possible improvement. y can be evaluated once and stored
        h_m = self.c_m * (self.x[0] * y[0])**(-1j * self.eta_m) \
            * utils._g_l_1(ell, z_ar)
        # TODO: possible improvement. _g_l can be evaluated once and stored

        Fy = irfft(np.conj(h_m)) * y**(-self.nu) * np.sqrt(np.pi) / 4.
        # here the ordering of N_extrap_begin and N_extrap_end is reversed
        # since we have moved to Fourier space
        return (
            y[self.N_extrap_end:self.N - self.N_extrap_begin],
            Fy[self.N_extrap_end:self.N - self.N_extrap_begin]
        )

    def fftlog_second_derivative(self, ell):
        r"""Performs the fftlog with the second derivative of the Bessel.

        Calculates

        .. math::
            F(y) = \int_0^\infty \frac{dx}{x} f(x) j''_{\rm \ell}(xy)

        where :math:`j_\ell` is the spherical Bessel function of order
        :math:`\ell`.

        Parameters
        ----------
        ell: int
            Order of the Bessel function present in the integral

        Returns
        -------
        y, Fy: numpy.ndarray, numpy.ndarray
         Logarithmically spaced array, set as
         ``y[:] = (ell+1)/x[::-1]``, FFTlog transform evaluated
         over the ``y`` array
        """
        z_ar = self.nu + 1j * self.eta_m
        y = (ell + 1.) / self.x[::-1]
        # TODO: possible improvement. y can be evaluated once and stored
        h_m = self.c_m * (self.x[0] * y[0])**(-1j * self.eta_m) \
            * utils._g_l_2(ell, z_ar)
        # TODO: possible improvement. _g_l can be evaluated once and stored

        Fy = irfft(np.conj(h_m)) * y**(-self.nu) * np.sqrt(np.pi) / 4.
        # here the ordering of N_extrap_begin and N_extrap_end is reversed
        # since we have moved to Fourier space
        return (
            y[self.N_extrap_end:self.N - self.N_extrap_begin],
            Fy[self.N_extrap_end:self.N - self.N_extrap_begin]
        )
