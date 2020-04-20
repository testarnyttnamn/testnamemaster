# -*- coding: utf-8 -*-
"""Cosmology

Class to store cosmological parameters and functions.
"""

import numpy as np


class CosmologyError(Exception):
    r"""
    Class to define Exception Error
    """
    pass


class Cosmology:
    """
    Class for GC spectroscopy observable
    """

    def __init__(self, cobaya_dic):
        """
        Parameters
        ----------
        cobaya_dic: dictionary
                    cosmological quantities from cobaya
        """
        self.cosmo_dic = cobaya_dic

    def growth_factor(self, zs, ks):
        """
        Calculates growth factor according as

        .. math::
                   D(z) &=\sqrt{P(z, k)/P(z=0, k)}\\

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
        D_z_k = self.cosmo_dic['Pk_delta'].P(zs, ks)
        D_z_k = np.sqrt(D_z_k / self.cosmo_dic['Pk_delta'].P(0.0001, ks))
        return D_z_k

    def growth_rate(self, zs, ks):
        """
        Calculates growth rate according as

        .. math::
                   f(z, k) &=-(1+z)dD(z, k)/dz\\

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
            f_z_k = -(1 + zs) * np.gradient(D_z_k)
            return f_z_k
        except CosmologyError:
            print('Check k is a value, not a list')

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
        try:
            self.cosmo_dic['D_z_k'] = self.growth_factor(zs, ks)
            self.cosmo_dic['f_z_k'] = self.growth_rate(zs, ks)
        except CosmologyError:
            print(
                r"Computation error in growth factor")
