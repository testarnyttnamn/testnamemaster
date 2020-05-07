# -*- coding: utf-8 -*-
"""Cosmology

Class to store cosmological parameters and functions.
"""

import numpy as np
from scipy import interpolate


class CosmologyError(Exception):
    r"""
    Class to define Exception Error
    """
    pass


class Cosmology:
    """
    Class for cosmological observables
    """

    def __init__(self):
        """
        Parameters
        ----------
        None

        """
        # (GCH): initialize cosmo dictionary
        # (ACD): Added speed of light to dictionary.!!!Important:it's in units
        # of km/s to be dimensionally consistent with H0.!!!!
        self.cosmo_dic = {'H0': 67.5,
                          'omch2': 0.122,
                          'ombh2': 0.022,
                          'mnu': 0.06,
                          'comov_dist': None,
                          'H': None,
                          'Pk_interpolator': None,
                          'Pk_delta': None,
                          'fsigma8': None,
                          'zk': None,
                          'b_gal': None,
                          'sigma_8': None,
                          'c': 3.0e5}

    def growth_factor(self, zs, ks):
        """
        Calculates growth factor according to

        .. math::
                   D(z, k) &=\sqrt{P(z, k)/P(z=0, k)}\\

        Parameters
        ----------
        zs: array
            list of redshift for the power spectrum
        ks: array
            list of modes for the power spectrum

        Returns
        -------
        growth factor

        """
        # GCH: Careful! This should be updated in the future!
        # we want to obtain delta directly from Cobaya
        # (in process)
        # This quantity depends on z and k
        try:
            D_z_k = self.cosmo_dic['Pk_delta'].P(zs, ks)
            D_z_k = np.sqrt(D_z_k / self.cosmo_dic['Pk_delta'].P(0.0, ks))
            return D_z_k
        except CosmologyError:
            print('Computation error in D(z, k)')

    def growth_rate(self, zs, ks):
        """
        Calculates growth rate according to

        .. math::
                   f(z, k) &=-(1+z)/D dD(z, k)/dz\\

        Parameters
        ----------
        zs: array
            list of redshift for the power spectrum
        ks: array
            list of modes for the power spectrum

        Returns
        -------
        growth rate

        """
        # GCH: Careful! This should be updated in the future!
        # we want to obtain delta directly from Cobaya
        # (in process)
        # This quantity depends on z and k
        # I assume 1+z=1/a where a: scale factor
        D_z_k = self.growth_factor(zs, ks)
        # This will work when k is fixed, not an array
        try:
            f_z_k = -(1 + zs) * np.gradient(D_z_k) / D_z_k
            return f_z_k
        except CosmologyError:
            print('Computation error in f(z, k)')
            print('ATTENTION: Check k is a value, not a list')

    def update_cosmo_dic(self, zs, ks):
        """
        Update the dictionary with other cosmological quantities

        Parameters
        ----------
        zs: array
            list of redshift for the power spectrum
        ks: array
            list of modes for the power spectrum

        Returns
        -------
        None

        """
        # (GCH): this function is superfluous
        # just in case we want to have always
        # an updated dictionary with D_z and f
        self.cosmo_dic['D_z_k'] = self.growth_factor(zs, ks)
        self.cosmo_dic['f_z_k'] = self.growth_rate(zs, ks)

    def interp_comoving_dist(self, zs):
        """
        Adds an interpolator for comoving distance to the dictionary so that
        it can be evaluated at redshifts not explictly supplied to cobaya.

        Parameters
        ----------
        zs: array
            list of redshift comoving distance is evaluated at.
        Returns
        -------
        None

        """
        self.cosmo_dic['r_z_func'] = interpolate.InterpolatedUnivariateSpline(
            x=zs, y=self.cosmo_dic['comov_dis'], ext=2)
