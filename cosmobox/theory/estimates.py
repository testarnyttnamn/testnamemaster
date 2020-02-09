# -*- coding: utf-8 -*-
"""Estimates

Module for theory estimates of observables.
"""

import numpy as np
from scipy import integrate
from scipy import interpolate


class ShearShear:
    """
    Class for weak lensing observables.
    """
    def __init__(self, bin_i, bin_j, n_type='istf', n_file=None, **kwargs):
        """
        Parameters
        ----------
        bin_i: array-like, floats
               Redshift bounds of bin i (lower, higher)
        bin_j: array-like, floats
               Redshift bounds of bin j (lower, higher)
        n_type: string
                Parameter to specify which n(z) to use:
                'istf' - Use Euclid IST:Forecasting n(z)
                'custom' - Input custom n(z). In this case, n_file must also
                           be specified.
        n_file: string
                Location of custom n(z) file.

        """
        if n_type == 'istf':
            self.n_i = self.p_up(bin_i[0], bin_i[1])
            self.n_j = self.p_up(bin_j[0], bin_j[1])
        elif n_type == 'custom':
            if n_file is None:
                raise Exception('When choosing a custom n(z), you must'
                                'specify the file location using n_file.')

    def n_istf(self, z):
        """
        Implements true galaxy source distribution as defined by Euclid
        IST: Forecasting.

        See Eq. 113 of https://arxiv.org/abs/1910.09273

        Parameters
        ----------
        z: float
           Redshift at which to evaluate distribution.

        Returns
        -------
        float
            n(z) for ISTF source distribution.
        """
        zm = 0.9
        z0 = zm / np.sqrt(2.0)
        n_val = ((z / z0)**2.0) * np.exp((-(z / z0)**1.5))
        return n_val

    def p_phot(self, zp, z):
        """
        Probability distribution function, describing the probability that
        a galaxy with redshift z has a measured redshift zp.

        Eq. 115 of https://arxiv.org/abs/1910.09273

        Parameters
        ----------
        zp: float
            Measured photometric redshift
        z: float
           True redshift

        Returns
        -------
        float
            Probability.
        """
        cb = 1.0
        zb = 0.0
        sigb = 0.05
        c0 = 1.0
        z0 = 0.1
        sig0 = 0.05
        fout = 0.1

        fac1 = (1.0 - fout) / (np.sqrt(2.0 * np.pi) * sigb * (1.0 + z))
        fac2 = fout / (np.sqrt(2.0 * np.pi) * sig0 * (1.0 + z))

        p_val = ((fac1 * np.exp((-0.5) * ((z - (cb * zp) -
                                           zb) / (sigb * (1.0 + z)))**2.0)) +
                 (fac2 * np.exp((-0.5) * ((z - (c0 * zp) -
                                           z0) / (sig0 * (1.0 + z)))**2.0)))

        return p_val

    def _n_phot_int(self, z, bin_z_max, bin_z_min):
        """
        Defines integrand to calculate denominator of Eq. 112 in
        https://arxiv.org/abs/1910.09273

        Parameters
        ----------
        z: float
           Redshift
        bin_z_max: float
                   Upper limit of bin
        bin_z_min: float
                   Lower limit of bin

        Returns
        -------
        float
            Denominator integrand value.
        """
        ret_val = self.n_istf(z) * integrate.quad(self.p_phot, a=bin_z_min,
                                                  b=bin_z_max, args=(z))[0]
        return ret_val

    def p_up(self, bin_z_min, bin_z_max):
        """
        Computes the true galaxy distribution, by carrying out convolution of
        true n(z) with photometric uncertainty PDF.

        Eq. 112 of https://arxiv.org/abs/1910.09273

        Parameters
        ----------
        bin_z_max: float
                   High z end of bin
        bin_z_min: float
                   Low z end of bin

        Returns
        -------
        float
            Observed n(z) with photometric redshift uncertainties accounted.
        """
        z_list = np.arange(0.001, 2.5, 0.1)
        n_nums_list = []
        for zind in range(len(z_list)):
            finprod = (self.n_istf(z_list[zind]) *
                       integrate.quad(self.p_phot, a=bin_z_min,
                                      b=bin_z_max, args=(z_list[zind]))[0])
            n_nums_list.append(finprod)

        n_denom = integrate.quad(self._n_phot_int, a=z_list[0],
                                 b=z_list[-1], args=(bin_z_max, bin_z_min))[0]

        res = np.array(n_nums_list) / n_denom

        true_bin = interpolate.InterpolatedUnivariateSpline(z_list, res, ext=1)

        return true_bin
